from google.cloud import compute_v1
from google.oauth2 import service_account
import time
import os


PROJECT_ID = "mineros-blockchain"
ZONE = 'us-east1-b'
CREDENTIALS_PATH = os.environ.get("CREDENTIALS_PATH")


def create_multiple_instances(num_instances: int) -> None:
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH)

    # Configuración de la instancia
    INSTANCE_NAME_PREFIX = 'miner-cpu'  # Nombre de la instancia que crearás
    MACHINE_TYPE = f'zones/{ZONE}/machineTypes/e2-highcpu-4'
    SUBNETWORK = f'projects/{PROJECT_ID}/regions/us-east1/subnetworks/default'
    SOURCE_IMAGE = f'projects/{PROJECT_ID}/global/images/pow-miner-1718748034'
    NETWORK_INTERFACE = {
        'subnetwork': SUBNETWORK,
        'access_configs': [
            {
                'name': 'External NAT'
            }
        ]
    }

    for i in range(num_instances):
        timestamp = int(time.time())

        instance_name = f"{INSTANCE_NAME_PREFIX}{round(timestamp)}"
        config = {
            'name': instance_name,
            'machine_type': MACHINE_TYPE,
            'disks': [
                {
                    'boot': True,
                    'auto_delete': True,
                    'initialize_params': {
                        'source_image': SOURCE_IMAGE,
                    }
                }
            ],
            'network_interfaces': [NETWORK_INTERFACE],
            'metadata': {
                'items': [
                    {
                        'key': 'startup-script',
                        'value': '#!/bin/bash\n'
                                 'sudo docker run -d -p 5000:5000 --name minero-cpu simondileogit/blockchain-miner:1.0.0'
                    }
                ]
            }
        }

        print(f"Creating instance {instance_name}...")
        compute_client = compute_v1.InstancesClient(credentials=credentials)

        # Crear la instancia
        compute_client.insert(
            project=PROJECT_ID,
            zone=ZONE,
            instance_resource=config
        )

    print(f"Instances created succesfully.")


def get_active_instance_count() -> int:
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH)

    compute_client = compute_v1.InstancesClient(credentials=credentials)

    # Listar todas las instancias en la zona especificada
    instance_list = compute_client.list(project=PROJECT_ID, zone=ZONE)

    active_instance_count = 0

    # Iterar sobre las instancias y contar cuántas están en estado "RUNNING"
    for instance in instance_list:
        if instance.status == 'RUNNING':
            active_instance_count += 1

    return active_instance_count


def destroy_all_instances() -> None:
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH)

    compute_client = compute_v1.InstancesClient(credentials=credentials)

    # Listar todas las instancias en la zona especificada
    instance_list = compute_client.list(project=PROJECT_ID, zone=ZONE)

    # Iterar sobre las instancias y eliminarlas una por una
    for instance in instance_list:
        instance_name = instance.name

        print(f"Deleting instance {instance_name}...")

        # Eliminar la instancia
        compute_client.delete(
            project=PROJECT_ID, zone=ZONE, instance=instance_name)

    print("All instances have been destroyed.")
