import os
import sys
import subprocess
import ast
import argparse
import time  # Importar el módulo time

GPU_MAX_RANGE = 85000

def parse_args():
    parser = argparse.ArgumentParser(description='Script para encontrar el nonce usando GPU')
    parser.add_argument('from_range', type=int, help='Rango inicial')
    parser.add_argument('to_range', type=int, help='Rango final')
    parser.add_argument('challenge', type=str, help='Desafío a resolver')
    parser.add_argument('block_content_hash', type=str, help='Hash del contenido del bloque')
    return parser.parse_args()

def dividir_rango(from_range, to_range, challenge, block_content_hash):
    current_dir = os.getcwd()
    block_hash = ""
    nonce = 0

    # Define relative paths
    src_dir = current_dir
    src_file = "find_nonce_gpu.cu"
    output_file = "find_nonce_gpu"
    result_file = "json_output.json"

    # Create the full paths
    src_path = os.path.join(src_dir, src_file)
    output_path = os.path.join(src_dir, output_file)
    result_path = os.path.join(current_dir, result_file)

    # Crea vacío el archivo de resultado
    with open(result_path, "w") as f:
        f.truncate(0)

    gpu_from = from_range
    gpu_to = gpu_from + GPU_MAX_RANGE
    nonce_found = False

    if not os.path.isfile(output_path):
        # Call nvcc to compile the CUDA file if the output file does not exist
        print(f"Compiling {src_path} to {output_path}")
        subprocess.call(["nvcc", src_path, "-o", output_path])
    
    end_time = None
    start_time = time.time()  # Captura el tiempo de inicio
    # Invocacion del cuda
    while (not nonce_found) and (gpu_from <= to_range):
        print(f"Running {output_path} with range {gpu_from} to {gpu_to}")
        subprocess.call(
            [output_path, str(gpu_from), str(gpu_to), str(challenge), block_content_hash], stdout=subprocess.DEVNULL)

        if not os.path.isfile(result_path):
            print(f"Result file {result_path} does not exist.", file=sys.stderr)
            break

        with open(result_path, "r") as file:
            result = file.readlines()
        
        if result:
            try:
                parsed_result = ast.literal_eval(result[0])
                block_hash = parsed_result["block_hash"]
                nonce = parsed_result["nonce"]
                if nonce > 0 and block_hash != "":
                    nonce_found = True
                    end_time = time.time()
            except (IndexError, SyntaxError, ValueError) as e:
                print(f"Error al analizar el resultado: {e}", file=sys.stderr)
                break
        else:
            print("El archivo de resultados está vacío o no se generó correctamente.", file=sys.stderr)
            break

        gpu_from = gpu_to + 1
        gpu_to += GPU_MAX_RANGE

        if gpu_to > to_range:
            gpu_to = to_range
        
        if (not end_time):
            end_time = time.time()

    # Captura el tiempo de finalización
    elapsed_time = end_time - start_time  # Calcula el tiempo transcurrido

    print(f"block_hash: {block_hash}", file=sys.stdout, flush=True)
    print(f"nonce: {nonce}", file=sys.stdout, flush=True)
    print(f"Tiempo transcurrido: {elapsed_time:.2f} segundos", file=sys.stdout, flush=True)

if __name__ == "__main__":
    args = parse_args()
    dividir_rango(args.from_range, args.to_range, args.challenge, args.block_content_hash)

# EJECUCION
# python3 server.py 0 100000000 00 hola
