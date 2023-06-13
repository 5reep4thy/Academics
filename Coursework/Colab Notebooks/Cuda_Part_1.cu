{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"name":"Cuda_Part_1.cu","provenance":[],"authorship_tag":"ABX9TyPKoJcahBit25QhDkdi3nYW"},"kernelspec":{"name":"python3","display_name":"Python 3"},"accelerator":"GPU"},"cells":[{"cell_type":"code","metadata":{"id":"RyPy8V8ze9qO","executionInfo":{"status":"ok","timestamp":1604398226450,"user_tz":-330,"elapsed":5454,"user":{"displayName":"SREEPATHY JAYANAND SREEPATHY JAYANAND","photoUrl":"","userId":"08275889955790806613"}},"outputId":"478e6471-0b16-4803-8abd-0b62cbbb74eb","colab":{"base_uri":"https://localhost:8080/"}},"source":["%%cu\n","#include<bits/stdc++.h>\n","using namespace std;\n"," __global__ void vecAdd( double *a,   double *b,  double *c, int n)\n","{\n","    int id = blockIdx.x*blockDim.x+threadIdx.x;\n","    int bs = blockDim.x;\n","    for (int k = id; k < n; k += bs) {\n","        c[k] = a[k] + b[k];\n","    }\n","    \n","}\n"," \n","int main( int argc, char* argv[] )\n","{\n","    int n = 10;\n","    \n","    \n","      double *h_a;\n","     double *h_b;\n","      double *h_c;\n"," \n","     double *d_a;\n","      double *d_b;\n","     double *d_c;\n"," \n","    size_t bytes = n*sizeof(  double);\n"," \n","    h_a = ( double*)malloc(bytes);\n","    h_b = (  double*)malloc(bytes);\n","    h_c = (  double*)malloc(bytes);\n"," \n","    cudaMalloc(&d_a, bytes);\n","    cudaMalloc(&d_b, bytes);\n","    cudaMalloc(&d_c, bytes);\n"," \n","    int i;\n","    for( i = 0; i < n; i++ ) {\n","        h_a[i] = i;\n","        h_b[i] = i;\n","    }\n"," \n","    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);\n","    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);\n"," \n","    int blockSize, gridSize;\n"," \n","    blockSize = 1;\n"," \n","    gridSize = 1;\n"," \n","    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);\n"," \n","    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );\n","    \n","      for(i=0; i<n; i++)\n","     printf(\"%f, \", h_a[i]);\n"," printf(\"\\n\");\n"," for(i=0; i<n; i++)\n","     printf(\"%f, \", h_b[i]);\n"," printf(\"\\n\");\n"," for(i=0; i<n; i++)\n","     printf(\"%f, \", h_c[i]);\n","    \n","    cudaFree(d_a);\n","    cudaFree(d_b);\n","    cudaFree(d_c);\n"," \n","    free(h_a);\n","    free(h_b);\n","    free(h_c);\n"," \n","    return 0;\n","}"],"execution_count":null,"outputs":[{"output_type":"stream","text":["0.000000, 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, \n","0.000000, 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, \n","0.000000, 2.000000, 4.000000, 6.000000, 8.000000, 10.000000, 12.000000, 14.000000, 16.000000, 18.000000, \n"],"name":"stdout"}]},{"cell_type":"code","metadata":{"id":"4wqMlLn9cU2v","executionInfo":{"status":"ok","timestamp":1604144611072,"user_tz":-330,"elapsed":5453,"user":{"displayName":"SREEPATHY JAYANAND SREEPATHY JAYANAND","photoUrl":"","userId":"08275889955790806613"}},"outputId":"efb4b1be-2452-465e-e54f-7e5d141ba9c3","colab":{"base_uri":"https://localhost:8080/"}},"source":["%%cu\n","#include<bits/stdc++.h>\n","using namespace std;\n"," __global__ void vecAdd( int *a, int *b,  int *c, int n)\n","{\n","    int id = blockIdx.x*blockDim.x+threadIdx.x;\n","    int bs = blockDim.x;\n","\n","\n","    for (int k = id; k <= n * n - 1; k += bs) {\n","        int tempr = k / n;\n","        int tempc = k % n;\n","        if (tempr >= n)\n","            break;\n","        for (int p = 0; p < n; p++) {\n","            c[tempr * n + tempc] += a[n * tempr + p] * b[tempc + p * n];\n","        }\n","    }\n","\n","}\n"," \n","int main( int argc, char* argv[] )\n","{\n","    int n = 4;\n"," \n","    \n","      int *h_a;\n","     int *h_b;\n","      int *h_c;\n"," \n","     int *d_a;\n","      int *d_b;\n","     int *d_c;\n"," \n","    size_t bytes = n * n * sizeof(int);\n"," \n","    h_a = (int*)malloc(bytes);\n","    h_b = (int*)malloc(bytes);\n","    h_c = (int*)malloc(bytes);\n"," \n","    cudaMalloc(&d_a, bytes);\n","    cudaMalloc(&d_b, bytes);\n","    cudaMalloc(&d_c, bytes);\n"," \n","    int i;\n","    for (int j = 0; j < n; j++)\n","        for ( i = 0; i < n; i++ ) {\n","            h_a[i + n * j] = i;\n","            h_b[i + n * j] = i;\n","        }\n"," \n","    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);\n","    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);\n"," \n","    int blockSize, gridSize;\n"," \n","    blockSize = 2;\n"," \n","    gridSize = 1;\n"," \n","    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);\n"," \n","    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );\n","    \n","\n","    for (int i = 0; i < n; i++){\n","        for (int j = 0; j < n; j++) {\n","            printf(\"%d \", h_a[i * n + j]);\n","        }\n","        printf(\"\\n\");\n","     }\n"," printf(\"\\n\");\n","    for (int i = 0; i < n; i++){\n","        for (int j = 0; j < n; j++) {\n","            printf(\"%d \", h_b[i * n + j]);\n","        }\n","        printf(\"\\n\");\n","     }\n"," printf(\"\\n\");\n","     for (int i = 0; i < n; i++){\n","        for (int j = 0; j < n; j++) {\n","            printf(\"%d \", h_c[i * n + j]);\n","        }\n","        printf(\"\\n\");\n","     }\n","    \n","    cudaFree(d_a);\n","    cudaFree(d_b);\n","    cudaFree(d_c);\n"," \n","    free(h_a);\n","    free(h_b);\n","    free(h_c);\n"," \n","    return 0;\n","}"],"execution_count":null,"outputs":[{"output_type":"stream","text":["0 1 2 3 \n","0 1 2 3 \n","0 1 2 3 \n","0 1 2 3 \n","\n","0 1 2 3 \n","0 1 2 3 \n","0 1 2 3 \n","0 1 2 3 \n","\n","0 6 12 18 \n","0 6 12 18 \n","0 6 12 18 \n","0 6 12 18 \n","\n"],"name":"stdout"}]},{"cell_type":"code","metadata":{"id":"ADdfLk6cc-dP"},"source":["!apt-get --purge remove cuda nvidia* libnvidia-*\n","!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge\n","!apt-get remove cuda-*\n","!apt autoremove\n","!apt-get update"],"execution_count":null,"outputs":[]},{"cell_type":"code","metadata":{"id":"soBQOicQfGo7"},"source":["!wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb\n","!dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb\n","!apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub\n","!apt-get update\n","!apt-get install cuda-9.2"],"execution_count":null,"outputs":[]},{"cell_type":"code","metadata":{"id":"FCV6Pi_0cFzr","executionInfo":{"status":"ok","timestamp":1605617731644,"user_tz":-330,"elapsed":7038,"user":{"displayName":"SREEPATHY JAYANAND SREEPATHY JAYANAND","photoUrl":"","userId":"08275889955790806613"}},"outputId":"4d1ae0da-757f-42fc-94ec-dea9c5a61334","colab":{"base_uri":"https://localhost:8080/"}},"source":["!pip install git+git://github.com/andreinechaev/nvcc4jupyter.git"],"execution_count":1,"outputs":[{"output_type":"stream","text":["Collecting git+git://github.com/andreinechaev/nvcc4jupyter.git\n","  Cloning git://github.com/andreinechaev/nvcc4jupyter.git to /tmp/pip-req-build-jidj_trd\n","  Running command git clone -q git://github.com/andreinechaev/nvcc4jupyter.git /tmp/pip-req-build-jidj_trd\n","Building wheels for collected packages: NVCCPlugin\n","  Building wheel for NVCCPlugin (setup.py) ... \u001b[?25l\u001b[?25hdone\n","  Created wheel for NVCCPlugin: filename=NVCCPlugin-0.0.2-cp36-none-any.whl size=4307 sha256=14abf0e8244b1124ebec44d0b07341a9a834cabc2bc1bc637ec0dc42f7d69b0a\n","  Stored in directory: /tmp/pip-ephem-wheel-cache-_btpoymx/wheels/10/c2/05/ca241da37bff77d60d31a9174f988109c61ba989e4d4650516\n","Successfully built NVCCPlugin\n","Installing collected packages: NVCCPlugin\n","Successfully installed NVCCPlugin-0.0.2\n"],"name":"stdout"}]},{"cell_type":"code","metadata":{"id":"J1jcdmaWcKUR","executionInfo":{"status":"ok","timestamp":1605617734393,"user_tz":-330,"elapsed":1305,"user":{"displayName":"SREEPATHY JAYANAND SREEPATHY JAYANAND","photoUrl":"","userId":"08275889955790806613"}},"outputId":"58d1e9f7-06de-42d6-ddff-f598f22f66a1","colab":{"base_uri":"https://localhost:8080/"}},"source":["%load_ext nvcc_plugin"],"execution_count":2,"outputs":[{"output_type":"stream","text":["created output directory at /content/src\n","Out bin /content/result.out\n"],"name":"stdout"}]}]}