#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

using namespace std;


cl_context createContext(cl_device_id* device_id, cl_int* ret)
{
    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, device_id, NULL, NULL, ret);
    printf("ret at %d is %d\n", __LINE__, ret);
    return context;
}


tuple<char*, size_t> loadKernel(char* fn) {
    // Load the kernel source code into the array source_str
    FILE* fp;
    char* source_str;
    size_t source_size;

    fp = fopen(fn, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    printf("kernel loading done\n");

    return { source_str, source_size };
}

tuple<cl_device_id, cl_int> getDeviceAndPlatform() {
    // Get platform and device information
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;


    cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    cl_platform_id* platforms = NULL;
    platforms = (cl_platform_id*)malloc(ret_num_platforms * sizeof(cl_platform_id));

    ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
    printf("ret at %d is %d\n", __LINE__, ret);
    return { device_id, ret };
}

int convolution3DForward() {
    typedef float numeric_type;
    const int IMAGE_C = 3;
    const int IMAGE_W = 128;
    const int IMAGE_H = 128;
    const int FILTER_W = 4;
    const int FILTER_C = 3;
    const int OUTPUT_C = (IMAGE_C - FILTER_C) + 1;
    const int OUTPUT_W = IMAGE_W - FILTER_W + 1;
    const int OUTPUT_H = IMAGE_H - FILTER_W + 1;
    const int IMAGE_SIZE = IMAGE_H * IMAGE_W * IMAGE_C;
    const int FILTER_SIZE = FILTER_W * FILTER_W * FILTER_C;
    const int OUTPUT_SIZE = OUTPUT_C * OUTPUT_W * OUTPUT_H;
    numeric_type* image = (numeric_type*)malloc(sizeof(numeric_type) * IMAGE_SIZE);
    numeric_type* filter = (numeric_type*)malloc(sizeof(numeric_type) * FILTER_SIZE);

    for (int i = 0; i < IMAGE_SIZE; i++) {
        image[i] = i % 2;
    }

    for (int i = 0; i < FILTER_SIZE; i++) {
        filter[i] = i %2;
    }
    // Load the kernel source code into the array source_str
    char* source_str;
    size_t source_size;
    auto kernalInfo = loadKernel("convolute3DForward.cl");
    source_str = get<0>(kernalInfo);
    source_size = get<1>(kernalInfo);

    // Get platform and device information
    auto deviceAndPlatform = getDeviceAndPlatform();
    cl_device_id device_id = get<0>(deviceAndPlatform);
    cl_int ret = get<1>(deviceAndPlatform);

    // Create an OpenCL context
    cl_context context = createContext(&device_id, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Create memory buffers on the device for each vector 
    cl_mem image_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        IMAGE_SIZE * sizeof(numeric_type), NULL, &ret);
    cl_mem filter_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        FILTER_SIZE * sizeof(numeric_type), NULL, &ret);
    cl_mem output_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        OUTPUT_SIZE * sizeof(numeric_type), NULL, &ret);

    // Copy the lists image and filter to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, image_mem_obj, CL_TRUE, 0,
        IMAGE_SIZE * sizeof(numeric_type), image, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clEnqueueWriteBuffer(command_queue, filter_mem_obj, CL_TRUE, 0,
        FILTER_SIZE * sizeof(numeric_type), filter, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    printf("before building\n");
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    printf("after building\n");
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "convolute3DForward", &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&image_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&filter_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel, 3, sizeof(int), &IMAGE_C);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &IMAGE_H);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 5, sizeof(int), &IMAGE_W);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 6, sizeof(int), &FILTER_C);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 7, sizeof(int), &FILTER_W);
    printf("ret at %d is %d\n", __LINE__, ret);

    //added this to fix garbage output problem
    //ret = clSetKernelArg(kernel, 3, sizeof(int), &LIST_SIZE);

    printf("before execution\n");
    // Execute the OpenCL kernel on the list
    size_t global_item_size = OUTPUT_SIZE; // Process the entire lists
    size_t local_item_size = 25; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_item_size, &local_item_size, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);
    printf("after execution\n");
    // Read the memory buffer output on the device to the local variable output
    numeric_type* output = (numeric_type*)malloc(sizeof(numeric_type) * OUTPUT_SIZE);
    ret = clEnqueueReadBuffer(command_queue, output_mem_obj, CL_TRUE, 0,
        OUTPUT_SIZE * sizeof(numeric_type), output, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);
    printf("after copying\n");
    // Display the result to the screen
    for (int i = 0; i < OUTPUT_SIZE; i++)
        printf("%f\n", output[i]);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(image_mem_obj);
    ret = clReleaseMemObject(filter_mem_obj);
    ret = clReleaseMemObject(output_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(image);
    free(filter);
    free(output);
    return 0;


}

int convolution3DBackward() {
    typedef float numeric_type;
    const int IMAGE_C = 3;
    const int IMAGE_W = 128;
    const int IMAGE_H = 128;
    const int FILTER_W = 4;
    const int FILTER_C = 3;
    const int OUTPUT_C = (IMAGE_C - FILTER_C) + 1;
    const int OUTPUT_W = IMAGE_W - FILTER_W + 1;
    const int OUTPUT_H = IMAGE_H - FILTER_W + 1;
    const int IMAGE_SIZE = IMAGE_H * IMAGE_W * IMAGE_C;
    const int FILTER_SIZE = FILTER_W * FILTER_W * FILTER_C;
    const int OUTPUT_SIZE = OUTPUT_C * OUTPUT_W * OUTPUT_H;
    numeric_type* image = (numeric_type*)malloc(sizeof(numeric_type) * IMAGE_SIZE);
    numeric_type* dOutput= (numeric_type*)malloc(sizeof(numeric_type) * OUTPUT_SIZE);
    numeric_type* filter = (numeric_type*)malloc(sizeof(numeric_type) * FILTER_SIZE);


    for (int i = 0; i < IMAGE_SIZE; i++) {
        image[i] = i % 2;
    }

    for (int i = 0; i < FILTER_SIZE; i++) {
        filter[i] = i % 2;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        dOutput[i] = i % 2;
    }

    // Load the kernel source code into the array source_str
    char* source_str;
    size_t source_size;
    auto kernalInfo = loadKernel("convolute3DBackward.cl");
    source_str = get<0>(kernalInfo);
    source_size = get<1>(kernalInfo);

    // Get platform and device information
    auto deviceAndPlatform = getDeviceAndPlatform();
    cl_device_id device_id = get<0>(deviceAndPlatform);
    cl_int ret = get<1>(deviceAndPlatform);

    // Create an OpenCL context
    cl_context context = createContext(&device_id, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Create memory buffers on the device for each vector 
    cl_mem image_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        IMAGE_SIZE * sizeof(numeric_type), NULL, &ret);
    cl_mem doutput_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        OUTPUT_SIZE * sizeof(numeric_type), NULL, &ret);
    cl_mem filter_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        FILTER_SIZE * sizeof(numeric_type), NULL, &ret);
    cl_mem dinput_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        IMAGE_SIZE * sizeof(numeric_type), NULL, &ret);
    cl_mem dfilter_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        FILTER_SIZE * sizeof(numeric_type), NULL, &ret);

    // Copy the lists image and filter to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, image_mem_obj, CL_TRUE, 0,
        IMAGE_SIZE * sizeof(numeric_type), image, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clEnqueueWriteBuffer(command_queue, doutput_mem_obj, CL_TRUE, 0,
        OUTPUT_SIZE * sizeof(numeric_type), filter, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, filter_mem_obj, CL_TRUE, 0,
        FILTER_SIZE * sizeof(numeric_type), filter, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    printf("before building\n");
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    printf("after building\n");
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "convolute3DBackward", &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&image_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&doutput_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&filter_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&dinput_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&dfilter_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel, 5, sizeof(int), &IMAGE_C);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 6, sizeof(int), &IMAGE_H);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 7, sizeof(int), &IMAGE_W);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 8, sizeof(int), &FILTER_C);
    printf("ret at %d is %d\n", __LINE__, ret);
    ret = clSetKernelArg(kernel, 9, sizeof(int), &FILTER_W);
    printf("ret at %d is %d\n", __LINE__, ret);

    //added this to fix garbage output problem
    //ret = clSetKernelArg(kernel, 3, sizeof(int), &LIST_SIZE);

    printf("before execution\n");
    // Execute the OpenCL kernel on the list
    size_t global_item_size = OUTPUT_SIZE; // Process the entire lists
    size_t local_item_size = 25; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_item_size, &local_item_size, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);
    printf("after execution\n");
    // Read the memory buffer output on the device to the local variable output
    numeric_type* dImage = (numeric_type*)malloc(sizeof(numeric_type) * IMAGE_SIZE);
    numeric_type* dFilter = (numeric_type*)malloc(sizeof(numeric_type) * FILTER_SIZE);
    ret = clEnqueueReadBuffer(command_queue, dinput_mem_obj, CL_TRUE, 0,
        IMAGE_SIZE * sizeof(numeric_type), dImage, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, dfilter_mem_obj, CL_TRUE, 0,
        FILTER_SIZE * sizeof(numeric_type), dFilter, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);
    printf("after copying\n");
    // Display the result to the screen
    for (int i = 0; i < IMAGE_SIZE; i++)
        printf("%f\n", dImage[i]);
    printf("***************");
    for (int i = 0; i < FILTER_SIZE; i++)
        printf("%f\n", dFilter[i]);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(image_mem_obj);
    ret = clReleaseMemObject(filter_mem_obj);
    ret = clReleaseMemObject(doutput_mem_obj);
    ret = clReleaseMemObject(dfilter_mem_obj);
    ret = clReleaseMemObject(dinput_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(image);
    free(filter);
    free(dOutput);
    free(dFilter);
    free(dImage);
    return 0;


}

int vector_add() {
    printf("started running\n");

    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int* A = (int*)malloc(sizeof(int) * LIST_SIZE);
    int* B = (int*)malloc(sizeof(int) * LIST_SIZE);
    for (i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    // Load the kernel source code into the array source_str
    char* source_str;
    size_t source_size;
    auto kernalInfo = loadKernel("vector_add_kernel.cl");
    source_str = get<0>(kernalInfo);
    source_size = get<1>(kernalInfo);

    // Get platform and device information
    auto deviceAndPlatform = getDeviceAndPlatform();
    cl_device_id device_id = get<0>(deviceAndPlatform);
    cl_int ret = get<1>(deviceAndPlatform);

    // Create an OpenCL context
    cl_context context = createContext(&device_id, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        LIST_SIZE * sizeof(int), NULL, &ret);



    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
        LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
        LIST_SIZE * sizeof(int), B, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    printf("before building\n");
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    printf("after building\n");
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
    printf("ret at %d is %d\n", __LINE__, ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);

    //added this to fix garbage output problem
    //ret = clSetKernelArg(kernel, 3, sizeof(int), &LIST_SIZE);

    printf("before execution\n");
    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_item_size, &local_item_size, 0, NULL, NULL);
    printf("after execution\n");
    // Read the memory buffer C on the device to the local variable C
    int* C = (int*)malloc(sizeof(int) * LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
        LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
    printf("after copying\n");
    // Display the result to the screen
    for (i = 0; i < LIST_SIZE; i++)
        printf("%d + %d = %d\n", A[i], B[i], C[i]);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    return 0;

}
int main(void) {
    convolution3DForward();
    vector_add();
}