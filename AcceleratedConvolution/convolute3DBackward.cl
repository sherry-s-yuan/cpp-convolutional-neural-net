__kernel void convolute3DBackward(
	const __global float * input, 
	const __global float * dOutput,
	const __global float * filter,
	__global float * dInput,
	__global float * dFilter,
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
				dInput[curInd + offset] = dOutput[curInd] * filter[ fIndex ];
				dFilter[ fIndex ] = dOutput[curInd] * input[curInd + offset];
				fIndex ++;
			}
		}
	}
	
}