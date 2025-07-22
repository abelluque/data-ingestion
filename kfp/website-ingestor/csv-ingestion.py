import os
from typing import List, NamedTuple

import kfp
from kfp import dsl, kubernetes
from kfp.dsl import Artifact, Input, Output


# Definición del componente ingest_csv_data para MinIO
@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2", "boto3==1.34.128"], # Añadir boto3
)
def ingest_csv_data(
    s3_bucket: str,
    s3_key: str,
    output_json_artifact: Output[Artifact],
    minio_endpoint: str, # Parámetro para la URL del endpoint de MinIO
    minio_access_key: str, # Parámetro para la clave de acceso de MinIO
    minio_secret_key: str # Parámetro para la clave secreta de MinIO
):
    """
    Descarga un archivo CSV desde S3 (o MinIO), selecciona las columnas relevantes y
    guarda los datos como un artefacto JSON.

    Args:
        s3_bucket (str): El nombre del bucket S3.
        s3_key (str): La clave (ruta) del archivo CSV dentro del bucket S3.
        output_json_artifact (Output[Artifact]): La ruta donde se guardará el JSON de datos procesados.
        minio_endpoint (str): La URL del endpoint de tu instancia de MinIO (ej. 'http://your-minio-endpoint:9000').
        minio_access_key (str): La clave de acceso para autenticarse con MinIO.
        minio_secret_key (str): La clave secreta para autenticarse con MinIO.
    """
    import pandas as pd
    import json
    import os
    import logging
    import boto3
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Crear un directorio temporal para el archivo descargado
    temp_dir = "/tmp/s3_data"
    os.makedirs(temp_dir, exist_ok=True)
    local_csv_path = os.path.join(temp_dir, os.path.basename(s3_key))

    logger.info(f"Intentando descargar {s3_key} de S3 bucket {s3_bucket} desde MinIO endpoint {minio_endpoint} a {local_csv_path}")

    try:
        # Configura el cliente boto3 para MinIO
        s3 = boto3.client(
            's3',
            endpoint_url=minio_endpoint,
            aws_access_key_id=minio_access_key,
            aws_secret_access_key=minio_secret_key,
            verify=False # Usar con precaución. Si usas HTTPS con certificados auto-firmados o no válidos.
                         # En producción, se recomienda configurar los certificados correctamente.
        )
        s3.download_file(s3_bucket, s3_key, local_csv_path)
        logger.info(f"Archivo descargado exitosamente de MinIO a: {local_csv_path}")
    except Exception as e:
        logger.error(f"Error al descargar el archivo de MinIO: {e}")
        raise RuntimeError(f"Error al descargar el archivo de MinIO: {e}")

    try:
        df = pd.read_csv(local_csv_path)
        logger.info(f"CSV leído exitosamente. Dimensiones del DataFrame: {df.shape}")
    except Exception as e:
        logger.error(f"Error al leer el archivo CSV local {local_csv_path}: {e}")
        raise RuntimeError(f"Error al leer el archivo CSV: {e}")

    relevant_columns = ["forma_ocurrencia", "etiqueta"]

    # Verificar si todas las columnas relevantes existen
    missing_columns = [col for col in relevant_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Faltan columnas requeridas en el CSV: {missing_columns}")
        raise ValueError(f"Las siguientes columnas requeridas no se encuentran en el archivo CSV: {missing_columns}")

    # Seleccionar solo las columnas relevantes
    df_selected = df[relevant_columns]

    # Convertir DataFrame a una lista de diccionarios
    processed_data = df_selected.to_dict(orient="records")

    # Guardar los datos procesados como un artefacto JSON
    with open(output_json_artifact.path, "w") as f:
        json.dump(processed_data, f, indent=4)
    logger.info(f"Datos procesados guardados en: {output_json_artifact.path}")


# Definición del componente process_and_store (sin cambios significativos, solo el origen del input)
@dsl.component(
    base_image="image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/minimal-gpu:2024.2",
    packages_to_install=[
        "langchain-community==0.3.8",
        "langchain==0.3.8",
        "sentence-transformers==2.4.0",
        "einops==0.7.0",
        "elastic-transport==8.15.1",
        "elasticsearch==8.16.0",
        "langchain-elasticsearch==0.3.0",
        "pandas==2.2.2",
    ],
)
def process_and_store(input_json_artifact: Input[Artifact], index_name: str):
    from elasticsearch import Elasticsearch
    import os
    import logging
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_elasticsearch import ElasticsearchStore
    import json

    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Función para obtener el cliente de Elasticsearch (sin cambios)
    def get_es_client():
        logger.info("Obteniendo cliente de Elasticsearch...")
        es_user = os.environ.get("ES_USER")
        es_pass = os.environ.get("ES_PASS")
        es_host = os.environ.get("ES_HOST")

        if not es_user or not es_pass or not es_host:
            print("Configuración de Elasticsearch no presente. Verifique host, puerto y api_key")
            exit(1)
        es_client = Elasticsearch(es_host,
                                  basic_auth=(es_user, es_pass),
                                  request_timeout=30,
                                  verify_certs=False)
        print(f"Estado del cliente de Elastic: {es_client.health_report()}")
        return es_client

    # Función para crear el índice (modificada para añadir mappings específicos de columnas)
    def create_index(es_client: Elasticsearch, index_name: str, mappings: dict = None):
        try:
            if es_client.indices.exists(index=index_name):
                logger.info(f"El índice '{index_name}' ya existe. Saltando la creación.")
                return False
            if mappings is None:
                mappings = {
                    "mappings": {
                        "properties": {
                            "page_content": {"type": "text"},
                            "metadata": {"type": "object"},
                            "forma_ocurrencia": {"type": "keyword"},
                            "etiqueta": {"type": "keyword"}
                        }
                    }
                }
            es_client.indices.create(index=index_name, body=mappings)
            logger.info(f"Índice '{index_name}' creado exitosamente.")
            return True
        except Exception as e:
            logger.error(f"Error al crear el índice '{index_name}': {e}")
            raise

    # Función para ingestar datos en Elasticsearch (sin cambios en la lógica de ingestión, solo en la preparación de documentos)
    def ingest(index_name, splits, es_client):
        """Ingesta documentos en Elasticsearch."""
        logger.info(f"Iniciando ingesta para el índice: {index_name}")
        try:
            model_kwargs = {"trust_remote_code": True}
            embeddings = HuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v1",
                model_kwargs=model_kwargs,
                show_progress=True,
            )

            db = ElasticsearchStore(
                index_name=index_name.lower(),
                embedding=embeddings,
                es_connection=es_client,
            )
            logger.info(f"Insertando datos en la base de datos")
            db.add_documents(splits)
            logger.info(f"Documentos subidos exitosamente a {index_name}")
        except Exception as e:
            logger.error(f"Error durante la ingesta para el índice {index_name}: {e}")


    """Procesa el contenido CSV y lo añade al almacén de Elasticsearch."""
    logger.info(f"Iniciando procesamiento para datos CSV.")
    es_client = get_es_client()

    # Asegurarse de que el índice exista
    create_index(es_client, index_name)

    # Leyendo el artefacto de la etapa anterior (datos JSON del CSV)
    with open(input_json_artifact.path, 'r') as input_file:
        processed_csv_data = json.load(input_file)

    if not processed_csv_data:
        logger.warning(f"No se encontraron datos en el CSV procesado. Saltando la ingesta.")
        return

    # Preparar los datos para la ingesta en Elasticsearch
    documents_for_es = []
    for row in processed_csv_data:
        # Combinar columnas relevantes en page_content para la incrustación
        page_content = f"Forma de Ocurrencia: {row.get('forma_ocurrencia', '')}. Etiqueta: {row.get('etiqueta', '')}"
        metadata = {
            "forma_ocurrencia": row.get("forma_ocurrencia"),
            "etiqueta": row.get("etiqueta"),
            "source": "minio_cleaned_data_processed.csv" # Actualizar metadato de origen para MinIO
        }
        documents_for_es.append(Document(page_content=page_content, metadata=metadata))

    # Ingestar datos en lotes a Elasticsearch
    ingest(index_name=index_name, splits=documents_for_es, es_client=es_client)

    logger.info(f"Procesamiento de datos CSV finalizado.")


@dsl.pipeline(name="Pipeline de Ingestión de Siniestros desde MinIO")
def siniestros_ingestion_pipeline_minio(
    s3_bucket: str = "your-minio-bucket-name", # Parámetro del bucket S3
    s3_key: str = "path/to/cleaned_data_processed.csv", # Parámetro de la clave S3
    minio_endpoint: str = "http://your-minio-service:9000", # Valor por defecto para MinIO
    minio_access_key: str = "minioadmin", # Valor por defecto, ¡reemplazar con un secreto!
    minio_secret_key: str = "minioadmin" # Valor por defecto, ¡reemplazar con un secreto!
):
    # Paso 1: Ingestar datos desde MinIO
    ingest_csv_task = ingest_csv_data(
        s3_bucket=s3_bucket,
        s3_key=s3_key,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key
    )

    # Paso 2: Procesar y almacenar los datos ingestados
    process_and_store_task = process_and_store(
        input_json_artifact=ingest_csv_task.outputs["output_json_artifact"],
        index_name="siniestros" # Nombre del índice fijo
    )

    # Aplicar acelerador y tolerancias (como en el pipeline original)
    process_and_store_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")
    kubernetes.add_toleration(process_and_store_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")

    # Establecer variables de entorno para Elasticsearch (como en el pipeline original)
    kubernetes.use_secret_as_env(
        process_and_store_task,
        secret_name="elasticsearch-es-elastic-user",
        secret_key_to_env={"elastic": "ES_PASS"},
    )
    process_and_store_task.set_env_variable("ES_HOST", "http://elasticsearch-es-http.composer-ai-apps.svc.cluster.local:9200")
    process_and_store_task.set_env_variable("ES_USER", "elastic")


if __name__ == "__main__":
    KUBEFLOW_ENDPOINT = os.getenv("KUBEFLOW_ENDPOINT")

    print(f"Conectando a kfp: {KUBEFLOW_ENDPOINT}")
    sa_token_path = "/run/secrets/kubernetes.io/serviceaccount/token" # noqa: S105
    if os.path.isfile(sa_token_path):
        with open(sa_token_path) as f:
            BEARER_TOKEN = f.read().rstrip()
    else:
        BEARER_TOKEN = os.getenv("BEARER_TOKEN")

    sa_ca_cert = "/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
    if os.path.isfile(sa_ca_cert):
        ssl_ca_cert = sa_ca_cert
    else:
        ssl_ca_cert = None

    client = kfp.Client(
        host=KUBEFLOW_ENDPOINT,
        existing_token=BEARER_TOKEN,
        ssl_ca_cert=None,
    )
    result = client.create_run_from_pipeline_func(
        siniestros_ingestion_pipeline_minio, # Usar el nuevo nombre de la función del pipeline
        experiment_name="siniestros_ingestion_from_minio", # Nombre del experimento actualizado
        enable_caching=False,
        arguments={
            "s3_bucket": "siniestros", 
            "s3_key": "cleaned_data_processed.csv", 
            "minio_endpoint": "https://minio-api-seguros-rivadavia.apps.cluster-sfw6q.sfw6q.sandbox29.opentlc.com", 
            "minio_access_key": "minio", 
            "minio_secret_key": "minio123" 
        }
    )
