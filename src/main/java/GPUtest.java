import jcuda.driver.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import static jcuda.driver.JCudaDriver.*;


public class GPUtest {

	public static void main(final String... args) throws Exception {
	
		 // Initialize the driver and create a context for the first device.
	    cuInit(0);
	    CUdevice device = new CUdevice();
	    cuDeviceGet(device, 0);
	    CUcontext context = new CUcontext();
	    cuCtxCreate(context, 0, device);
		
	    // Load the PTX that contains the kernel.
	    CUmodule module = new CUmodule();
	    cuModuleLoad(module, "vectorAdd.ptx");
		 // Obtain a handle to the kernel function.
		    CUfunction function = new CUfunction();
		    cuModuleGetFunction(function, module, "vectorAddTwo");
		int[] vectorA = new int[10000];
		int[] vectorB = new int[10000];
		
		
		int numElements = vectorA.length;
	    
	    CUdeviceptr device_vectorA = new CUdeviceptr();
	    cuMemAlloc(device_vectorA, numElements * Sizeof.INT);
	    cuMemcpyHtoD(device_vectorA, Pointer.to(vectorA),
	        numElements * Sizeof.INT);
	    CUdeviceptr device_vectorB = new CUdeviceptr();
	    cuMemAlloc(device_vectorB, numElements * Sizeof.INT);
	    cuMemcpyHtoD(device_vectorB, Pointer.to(vectorB),
	        numElements * Sizeof.INT);
	    
	 // Allocate the device input data, and copy the
	    // host input data to the device
	    
	    CUdeviceptr device_vectorOutput = new CUdeviceptr();
	    cuMemAlloc(device_vectorOutput, numElements * Sizeof.INT);
	 // Set up the kernel parameters: A pointer to an array
	    // of pointers which point to the actual values.
	    Pointer kernelParameters = Pointer.to(   
	    	Pointer.to(device_vectorA),
	        Pointer.to(device_vectorB),
	        Pointer.to(device_vectorOutput)
	    );
	    
	    int blockSizeX = 1;
	    int gridSizeY = 100;
	    int gridSizeX = 100;//(int)Math.ceil((double)numElements / blockSizeX);
	    cuLaunchKernel(function,
	        gridSizeX,  gridSizeY, 1,      // Grid dimension
	        blockSizeX, 1, 1,      // Block dimension
	        0, null,               // Shared memory size and stream
	        kernelParameters, null // Kernel- and extra parameters
	    );
	    
	    cuCtxSynchronize();
	    
	    int hostOutput[] = new int[numElements];
	    cuMemcpyDtoH(Pointer.to(hostOutput), device_vectorOutput,
	    		numElements * Sizeof.INT);
	    cuMemFree(device_vectorA);
	    cuMemFree(device_vectorB); 
	    cuMemFree(device_vectorOutput);
	    
	    
		/*
	short[] width           = {50};                                       // filter window width.
    int[] depth           = {1000};                                    // z.
    short framewidth      = 64;
    short frameheight     = 64;
    int N               = depth[0] * framewidth * frameheight;         // size of entry.
    float[] meanVector = generateTest(depth[0]);
    int[] test_Data = generateTest(N,depth[0]);
    
 // Initialize the driver and create a context for the first device.
    cuInit(0);
    CUdevice device = new CUdevice();
    cuDeviceGet(device, 0);
    CUcontext context = new CUcontext();
    cuCtxCreate(context, 0, device);
 // Load the PTX that contains the kernel.
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "medianFiltering.ptx");
 // Obtain a handle to the kernel function.
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "medianKernel");
    

    // Allocate the device input data, and copy the
    // host input data to the device
    int numElements = ((2 * width[0] + 1) * framewidth * frameheight);
    CUdeviceptr device_window = new CUdeviceptr();
    cuMemAlloc(device_window, numElements * Sizeof.FLOAT);

    numElements = 1;
    
    CUdeviceptr device_width = new CUdeviceptr();
    cuMemAlloc(device_width, numElements * Sizeof.SHORT);
    cuMemcpyHtoD(device_width, Pointer.to(width),
        numElements * Sizeof.SHORT);
    numElements = test_Data.length;
    
    CUdeviceptr device_test_Data = new CUdeviceptr();
    cuMemAlloc(device_test_Data, numElements * Sizeof.INT);
    cuMemcpyHtoD(device_test_Data, Pointer.to(test_Data),
        numElements * Sizeof.INT);
    numElements = 1;
    
    CUdeviceptr device_depth = new CUdeviceptr();
    cuMemAlloc(device_depth, numElements * Sizeof.INT);
    cuMemcpyHtoD(device_depth, Pointer.to(depth),
        numElements * Sizeof.INT);
    
    
    numElements = meanVector.length;   
    CUdeviceptr deviceMeanVector = new CUdeviceptr();
    cuMemAlloc(deviceMeanVector, numElements * Sizeof.FLOAT);
    cuMemcpyHtoD(deviceMeanVector, Pointer.to(meanVector),
        numElements * Sizeof.FLOAT);
    
    
    numElements = test_Data.length;
    // Allocate device output memory
    CUdeviceptr deviceOutput = new CUdeviceptr();
    cuMemAlloc(deviceOutput, numElements * Sizeof.FLOAT);

    // Set up the kernel parameters: A pointer to an array
    // of pointers which point to the actual values.
    Pointer kernelParameters = Pointer.to(   
    	Pointer.to(device_width),
        Pointer.to(device_window),
        Pointer.to(device_depth),
        Pointer.to(device_test_Data),
        Pointer.to(deviceOutput)
    );
    int blockSizeX = 1;
    int gridSizeY = 64;
    int gridSizeX = 64;//(int)Math.ceil((double)numElements / blockSizeX);
    cuLaunchKernel(function,
        gridSizeX,  gridSizeY, 1,      // Grid dimension
        blockSizeX, 1, 1,      // Block dimension
        0, null,               // Shared memory size and stream
        kernelParameters, null // Kernel- and extra parameters
    );
    cuCtxSynchronize();
    numElements = test_Data.length;
    float hostOutput[] = new float[numElements];
    cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
    		numElements * Sizeof.FLOAT);
    cuMemFree(device_width);
    cuMemFree(device_window);
    cuMemFree(device_depth);
    cuMemFree(device_test_Data);    
    cuMemFree(deviceOutput);
*/
    
	}
	public static float[] generateTest(int N)
	{
		float[] test = new float[N];
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
