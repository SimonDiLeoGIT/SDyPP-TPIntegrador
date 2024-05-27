#include <cstring>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

/* F, G and H are basic MD5 functions: selection, majority, parity */
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/* ROTATE_LEFT rotates x left n bits */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4 */
/* Rotation is separate from addition to prevent recomputation */
#define FF(a, b, c, d, x, s, ac) \
  {(a) += F ((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = ROTATE_LEFT ((a), (s)); \
    (a) += (b); \
  }
#define GG(a, b, c, d, x, s, ac) \
  {(a) += G ((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = ROTATE_LEFT ((a), (s)); \
    (a) += (b); \
  }
#define HH(a, b, c, d, x, s, ac) \
  {(a) += H ((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = ROTATE_LEFT ((a), (s)); \
    (a) += (b); \
  }
#define II(a, b, c, d, x, s, ac) \
  {(a) += I ((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = ROTATE_LEFT ((a), (s)); \
    (a) += (b); \
  }

__device__ inline void md5Hash(unsigned char* data, uint32_t length, uint32_t *a1, uint32_t *b1, uint32_t *c1, uint32_t *d1){
  const uint32_t a0 = 0x67452301;
  const uint32_t b0 = 0xEFCDAB89;
  const uint32_t c0 = 0x98BADCFE;
  const uint32_t d0 = 0x10325476;

  uint32_t a = 0;
  uint32_t b = 0;
  uint32_t c = 0;
  uint32_t d = 0;

  uint32_t vals[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  int i = 0;
  for(i=0; i < length; i++){
    vals[i / 4] |= data[i] << ((i % 4) * 8);
  }
  
  vals[i / 4] |= 0x80 << ((i % 4) * 8);

  uint32_t bitlen = length * 8;

  #define in0  (vals[0])//x
  #define in1  (vals[1])//y
  #define in2  (vals[2])//z
  #define in3  (vals[3])
  #define in4  (vals[4])
  #define in5  (vals[5])
  #define in6  (vals[6])
  #define in7  (vals[7])
  #define in8  (vals[8])
  #define in9  (vals[9])
  #define in10 (vals[10])
  #define in11 (vals[11])
  #define in12 (vals[12])
  #define in13 (vals[13])
  #define in14 (bitlen) //w = bit length
  #define in15 (0)

  //Initialize hash value for this chunk:
  a = a0;
  b = b0;
  c = c0;
  d = d0;

  /* Round 1 */
  #define S11 7
  #define S12 12
  #define S13 17
  #define S14 22
  FF ( a, b, c, d, in0,  S11, 3614090360); /* 1 */
  FF ( d, a, b, c, in1,  S12, 3905402710); /* 2 */
  FF ( c, d, a, b, in2,  S13,  606105819); /* 3 */
  FF ( b, c, d, a, in3,  S14, 3250441966); /* 4 */
  FF ( a, b, c, d, in4,  S11, 4118548399); /* 5 */
  FF ( d, a, b, c, in5,  S12, 1200080426); /* 6 */
  FF ( c, d, a, b, in6,  S13, 2821735955); /* 7 */
  FF ( b, c, d, a, in7,  S14, 4249261313); /* 8 */
  FF ( a, b, c, d, in8,  S11, 1770035416); /* 9 */
  FF ( d, a, b, c, in9,  S12, 2336552879); /* 10 */
  FF ( c, d, a, b, in10, S13, 4294925233); /* 11 */
  FF ( b, c, d, a, in11, S14, 2304563134); /* 12 */
  FF ( a, b, c, d, in12, S11, 1804603682); /* 13 */
  FF ( d, a, b, c, in13, S12, 4254626195); /* 14 */
  FF ( c, d, a, b, in14, S13, 2792965006); /* 15 */
  FF ( b, c, d, a, in15, S14, 1236535329); /* 16 */

  /* Round 2 */
  #define S21 5
  #define S22 9
  #define S23 14
  #define S24 20
  GG ( a, b, c, d, in1, S21, 4129170786); /* 17 */
  GG ( d, a, b, c, in6, S22, 3225465664); /* 18 */
  GG ( c, d, a, b, in11, S23,  643717713); /* 19 */
  GG ( b, c, d, a, in0, S24, 3921069994); /* 20 */
  GG ( a, b, c, d, in5, S21, 3593408605); /* 21 */
  GG ( d, a, b, c, in10, S22,   38016083); /* 22 */
  GG ( c, d, a, b, in15, S23, 3634488961); /* 23 */
  GG ( b, c, d, a, in4, S24, 3889429448); /* 24 */
  GG ( a, b, c, d, in9, S21,  568446438); /* 25 */
  GG ( d, a, b, c, in14, S22, 3275163606); /* 26 */
  GG ( c, d, a, b, in3, S23, 4107603335); /* 27 */
  GG ( b, c, d, a, in8, S24, 1163531501); /* 28 */
  GG ( a, b, c, d, in13, S21, 2850285829); /* 29 */
  GG ( d, a, b, c, in2, S22, 4243563512); /* 30 */
  GG ( c, d, a, b, in7, S23, 1735328473); /* 31 */
  GG ( b, c, d, a, in12, S24, 2368359562); /* 32 */

  /* Round 3 */
  #define S31 4
  #define S32 11
  #define S33 16
  #define S34 23
  HH ( a, b, c, d, in5, S31, 4294588738); /* 33 */
  HH ( d, a, b, c, in8, S32, 2272392833); /* 34 */
  HH ( c, d, a, b, in11, S33, 1839030562); /* 35 */
  HH ( b, c, d, a, in14, S34, 4259657740); /* 36 */
  HH ( a, b, c, d, in1, S31, 2763975236); /* 37 */
  HH ( d, a, b, c, in4, S32, 1272893353); /* 38 */
  HH ( c, d, a, b, in7, S33, 4139469664); /* 39 */
  HH ( b, c, d, a, in10, S34, 3200236656); /* 40 */
  HH ( a, b, c, d, in13, S31,  681279174); /* 41 */
  HH ( d, a, b, c, in0, S32, 3936430074); /* 42 */
  HH ( c, d, a, b, in3, S33, 3572445317); /* 43 */
  HH ( b, c, d, a, in6, S34,   76029189); /* 44 */
  HH ( a, b, c, d, in9, S31, 3654602809); /* 45 */
  HH ( d, a, b, c, in12, S32, 3873151461); /* 46 */
  HH ( c, d, a, b, in15, S33,  530742520); /* 47 */
  HH ( b, c, d, a, in2, S34, 3299628645); /* 48 */

  /* Round 4 */
  #define S41 6
  #define S42 10
  #define S43 15
  #define S44 21
  II ( a, b, c, d, in0, S41, 4096336452); /* 49 */
  II ( d, a, b, c, in7, S42, 1126891415); /* 50 */
  II ( c, d, a, b, in14, S43, 2878612391); /* 51 */
  II ( b, c, d, a, in5, S44, 4237533241); /* 52 */
  II ( a, b, c, d, in12, S41, 1700485571); /* 53 */
  II ( d, a, b, c, in3, S42, 2399980690); /* 54 */
  II ( c, d, a, b, in10, S43, 4293915773); /* 55 */
  II ( b, c, d, a, in1, S44, 2240044497); /* 56 */
  II ( a, b, c, d, in8, S41, 1873313359); /* 57 */
  II ( d, a, b, c, in15, S42, 4264355552); /* 58 */
  II ( c, d, a, b, in6, S43, 2734768916); /* 59 */
  II ( b, c, d, a, in13, S44, 1309151649); /* 60 */
  II ( a, b, c, d, in4, S41, 4149444226); /* 61 */
  II ( d, a, b, c, in11, S42, 3174756917); /* 62 */
  II ( c, d, a, b, in2, S43,  718787259); /* 63 */
  II ( b, c, d, a, in9, S44, 3951481745); /* 64 */

  a += a0;
  b += b0;
  c += c0;
  d += d0;

  *a1 = a;
  *b1 = b;
  *c1 = c;
  *d1 = d;
}


__device__ void cuda_hexcopy(char *output, uint32_t value, int offset) {
    const char *hex_chars = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
        output[offset + i] = hex_chars[(value >> (28 - i * 4)) & 0xF];
    }
}

__device__ int cuda_strlen(const char *str) {
    int length = 0;
    while (str[length] != '\0') {
        length++;
    }
    return length;
}

__global__ void cuda_md5(char *challenge, char *previous, int nonce_max) {
    int bloque_tamanio = blockDim.x;  // 384
    int bloque_numero  = blockIdx.x;  // 0 .. n (dinámico según nonce_max)
    int thread_numero  = threadIdx.x; // 0 .. 383

    int nonce = bloque_numero * bloque_tamanio + thread_numero;

    if (nonce >= nonce_max) return;

    int length = 40; // 32 (previous) + 8 (nonce)
    char data[40];

    cuda_hexcopy(data, nonce, 0);    
    for (int i = 8; i < 40; i++) {
        data[i] = previous[i - 8];
    }

    uint32_t a1, b1, c1, d1;
    md5Hash((unsigned char*) data, length, &a1, &b1, &c1, &d1);

    char md5_hash[33];  // 32 caracteres + terminador nulo
    md5_hash[32] = '\0';

    cuda_hexcopy(md5_hash, a1,  0);
    cuda_hexcopy(md5_hash, b1,  8);
    cuda_hexcopy(md5_hash, c1, 16);
    cuda_hexcopy(md5_hash, d1, 24);

    // Imprimir todos los hashes generados para depuración
    printf("Nonce: %d, Hash: %s\n", nonce, md5_hash);

    int length_challenge = cuda_strlen(challenge);
    for (int i = 0; i < length_challenge; i++) {
        if (md5_hash[i] != challenge[i]) {
            return;
        }
    }

    printf("> nonce: %i\n", nonce);
    printf("> hash: %s\n", md5_hash);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: %s <challenge> <previous> <nonce_max>\n", argv[0]);
        return 1;
    }

    char *challenge = argv[1];
    char *previous = argv[2];
    int nonce_max = atoi(argv[3]);

    char *d_challenge;
    char *d_previous;
    cudaMalloc(&d_challenge, sizeof(char) * (strlen(challenge) + 1));
    cudaMalloc(&d_previous, sizeof(char) * (strlen(previous) + 1));

    cudaMemcpy(d_challenge, challenge, sizeof(char) * (strlen(challenge) + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_previous, previous, sizeof(char) * (strlen(previous) + 1), cudaMemcpyHostToDevice);

    int num_blocks = (nonce_max + 383) / 384;
    cuda_md5<<<num_blocks, 384>>>(d_challenge, d_previous, nonce_max);

    cudaDeviceSynchronize();

    cudaFree(d_challenge);
    cudaFree(d_previous);

    return 0;
}
