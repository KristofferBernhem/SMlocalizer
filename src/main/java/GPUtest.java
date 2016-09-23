import jcuda.driver.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import static jcuda.driver.JCudaDriver.*;


public class GPUtest {

	public static void main(final String... args) throws Exception {
	
		//JCudaDriver.setExceptionsEnabled(true);	 // debug
		
	int width           = 50;                                       // filter window width.
    int depth           = 10000;                                    // z.
    short framewidth    = 64;
    short frameheight   = 64;
    int N               = depth * framewidth * frameheight;         // size of entry.
    int[] meanVector 	= generateTest(depth);
    int[] testData 		= generateTest(N,depth);
    
 // Initialize the driver and create a context for the first device.
    cuInit(0);
    CUdevice device = new CUdevice();
    cuDeviceGet(device, 0);
    CUcontext context = new CUcontext();
    cuCtxCreate(context, 0, device);
 // Load the PTX that contains the kernel.
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "CUDAFYSOURCETEMP.ptx");
 // Obtain a handle to the kernel function.
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "medianKernel");
    long time = System.nanoTime();
    // Transfer data to device and allocate memory on device for swap and results vectors.
    CUdeviceptr device_window 		= CUDA.allocateOnDevice((2 * width + 1) * framewidth * frameheight);
    CUdeviceptr device_test_Data 	= CUDA.copyToDevice(testData);
    CUdeviceptr device_meanVector 	= CUDA.copyToDevice(meanVector);
    CUdeviceptr deviceOutput 		= CUDA.allocateOnDevice(testData.length);

    
    // Set up parameters for launching kernel.  
    int filteWindowLength 		= (2 * width + 1) * framewidth * frameheight;
    int testDataLength 			= testData.length;
    int meanVectorLength 		= meanVector.length;
    Pointer kernelParameters 	= Pointer.to(   
    	Pointer.to(new int[]{width}),
        Pointer.to(device_window),
        Pointer.to(new int[]{filteWindowLength}),
        Pointer.to(new int[]{depth}),
        Pointer.to(device_test_Data),
        Pointer.to(new int[]{testDataLength}),
        Pointer.to(device_meanVector),
        Pointer.to(new int[]{meanVectorLength}),
        Pointer.to(deviceOutput),
        Pointer.to(new int[]{testDataLength})
    );
    int blockSizeX 	= 1;
    int blockSizeY 	= 1;
    int gridSizeY 	= framewidth;
    int gridSizeX 	= frameheight;
    cuLaunchKernel(function,
        gridSizeX,  gridSizeY, 1, 	// Grid dimension
        blockSizeX, blockSizeY, 1,  // Block dimension
        0, null,               		// Shared memory size and stream
        kernelParameters, null 		// Kernel- and extra parameters
    );
    cuCtxSynchronize();
     
    // Pull data from device.
    int hostOutput[] = new int[testData.length];
    cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
    		testData.length * Sizeof.INT);
    time = System.nanoTime() - time;
    System.out.println((time /1E6) + " ms");
   // Free up memory allocation on device, housekeeping.
    cuMemFree(device_window);   
    cuMemFree(device_test_Data);    
    cuMemFree(deviceOutput);

    
	}
	public static int[] generateTest(int N)
	{
		int[] test = new int[N];
		for (int i = 0; i < test.length; i++)
		{
			test[i] = 1;
		}
		return test;
	}
	public static int[] generateTest(int N, int depth) // Generate test vector to easily see if median calculations are correct.
    {
		int[] test = new int[N];
		int count = 0;
        for (int i = 0; i < test.length; i++)
        {
            test[i] = count;
            count++;
            if (count == depth)
                count = 0;
        }
        return test;
    }

}
