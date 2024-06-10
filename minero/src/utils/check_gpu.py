import subprocess


def check_for_nvidia_smi():
    try:
        result = subprocess.run(
            ['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("Se encontró GPU NVIDIA")
            print(result.stdout.decode('utf-8'))
            return True
        else:
            print("No se encontró GPU NVIDIA")
            return False
    except FileNotFoundError:
        print("nvidia-smi no está instalado o no se encontró en el PATH.")
        return False
