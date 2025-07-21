import os
from typing import List, NamedTuple

import kfp
from kfp import dsl, kubernetes
from kfp.dsl import Artifact, Input, Output


# Definición del componente ingest_csv_data (copiado aquí para un ejemplo autocontenido)
@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2"],
)
def ingest_csv_data(csv_file_path: str, output_json_artifact: Output[Artifact]):
    """
    Lee un archivo CSV desde una ruta especificada dentro del contenedor,
    selecciona las columnas relevantes y guarda los datos como un artefacto JSON.

    Args:
        csv_file_path (str): La ruta al archivo CSV dentro del contenedor.
        output_json_artifact (Output[Artifact]): La ruta donde se guardará el JSON de datos procesados.
    """
    import pandas as pd
    import json
    import os
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"Intentando leer CSV desde: {csv_file_path}")

    if not os.path.exists(csv_file_path):
        logger.error(f"Archivo no encontrado: {csv_file_path}")
        raise FileNotFoundError(f"El archivo CSV no se encontró en: {csv_file_path}")

    try:
        df = pd.read_csv(csv_file_path)
        logger.info(f"CSV leído exitosamente. Dimensiones del DataFrame: {df.shape}")
    except Exception as e:
        logger.error(f"Error al leer el archivo CSV {csv_file_path}: {e}")
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


# Definición del componente process_and_store (modificado)
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
        "pandas==2.2.2", # Mantener pandas por si acaso, aunque ingest_csv_data ya lo usa.
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
                            "forma_ocurrencia": {"type": "keyword"}, # Mapeo específico para forma_ocurrencia
                            "etiqueta": {"type": "keyword"}          # Mapeo específico para etiqueta
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
            "source": "siniestros.csv" # Añadir metadato de origen
        }
        documents_for_es.append(Document(page_content=page_content, metadata=metadata))

    # Ingestar datos en lotes a Elasticsearch
    ingest(index_name=index_name, splits=documents_for_es, es_client=es_client)

    logger.info(f"Procesamiento de datos CSV finalizado.")


@dsl.pipeline(name="Pipeline de Ingestión de Siniestros desde CSV")
def siniestros_ingestion_pipeline():
    # Define la ruta al archivo CSV *dentro* del contenedor Docker
    # Esta ruta debe coincidir con donde el Dockerfile copia el CSV.
    csv_internal_path = "/data/cleaned_data_processed.csv"

    # Paso 1: Ingestar datos desde CSV
    ingest_csv_task = ingest_csv_data(csv_file_path=csv_internal_path)

    # Paso 2: Procesar y almacenar los datos ingestados
    # Pasar el artefacto JSON de salida de ingest_csv_data a process_and_store
    # El nombre del índice se establece en "siniestros" según lo solicitado
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
    process_and_store_task.set_env_variable("ES_HOST", "http://elasticsearch-es-http:9200")
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
        siniestros_ingestion_pipeline, # Usar el nuevo nombre de la función del pipeline
        experiment_name="siniestros_ingestion_from_csv", # Nombre del experimento actualizado
        enable_caching=False,
        arguments={} # Ya no se necesitan argumentos para la función del pipeline en sí
    )

    # Para compilar el pipeline localmente (descomentar si es necesario):
    # from kfp import compiler
    # compiler.Compiler().compile(siniestros_ingestion_pipeline, 'siniestros_pipeline.yaml')
