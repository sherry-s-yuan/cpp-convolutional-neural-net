__kernel void convolute3DForward(
	const __global float * input, 
	__global float * output,
	const __global float * filter, 
	int IMAGE_C,
	int IMAGE_H,
	int IMAGE_W,
	int FILTER_C,
	int FILTER_SIZE
)
{
	
	const int curInd = get_global_id(0);
	
	int fIndex = 0;
	float sum = 0;
	
	for (int d = 0; d < FILTER_C; d++){
		for (int r = 0; r < FILTER_SIZE; r++)
		{
			for (int c = 0; c < FILTER_SIZE; c++)
			{
				int channelOffset = d * IMAGE_W * IMAGE_H;
				int rowOffset = r * IMAGE_W;
				int colOffset = c;
				int offset = channelOffset + rowOffset + colOffset;	
				sum += input[ curInd + offset] * filter[ fIndex ]; 
				fIndex ++;
			}
		}
	}
	
	
	output[curInd] = sum;
}