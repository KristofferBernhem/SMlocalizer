import jcuda.driver.*;
import jcuda.Pointer;
import jcuda.Sizeof;

import static jcuda.driver.JCudaDriver.*;

import java.util.ArrayList;


public class GPUtest {

	public static void main(final String... args) throws Exception {
		double convCriteria = 1E-8;
	    int maxIterations = 1000;
	    
		// TODO look over kernel for errors. Change median from int to float. 
		 // Initialize the driver and create a context for the first device.
	    cuInit(0);
	    CUdevice device = new CUdevice();
	    cuDeviceGet(device, 0);
	    CUcontext context = new CUcontext();
	    cuCtxCreate(context, 0, device);
	 // Load the PTX that contains the kernel.
	    CUmodule module = new CUmodule();
	    cuModuleLoad(module, "gFit.ptx");
	 // Obtain a handle to the kernel function.
	    CUfunction function = new CUfunction();
	    cuModuleGetFunction(function, module, "gaussFitter");
	    int[] testdata ={ // slize 45 SingleBead2
				3888, 3984,  6192,   4192, 3664,  3472, 3136,
				6384, 8192,  12368, 12720, 6032,  5360, 3408, 
				6192, 13760, 21536, 20528, 9744,  6192, 2896,
				6416, 15968, 25600, 28080, 12288, 4496, 2400,
				4816, 11312, 15376, 14816, 8016,  4512, 3360,
				2944, 4688,  7168,   5648, 5824,  3456, 2912,
				2784, 3168,  4512,   4192, 3472,  2768, 2912
		};
	    
	    int Ch = 1;
	    int nFrames = 10;
	    int[] gWindow = {7,7};
	    int[] Center = {3,3};
	    int[] inputPixelSize = {100,100};
	    int[] totalGain = {100,100};
	    
	    	ArrayList<fitParameters> fitThese = new ArrayList<fitParameters>();
		for (int Frame = 1; Frame <= nFrames;Frame++)					// Loop over all frames.
			{
	

			
				int[] dataFit = new int[gWindow[Ch-1]*gWindow[Ch-1]];							// Container for data to be fitted.
				int[] Coord = {3,3};
				
				for (int j = 0; j < gWindow[Ch-1]*gWindow[Ch-1]; j++)
				{
					int x =  Coord[0] - Math.round((gWindow[Ch-1])/2) +  (j % gWindow[Ch-1]);
					int y =  Coord[1] - Math.round((gWindow[Ch-1])/2) +  (j / gWindow[Ch-1]);
					//dataFit[j] = inputArray[x][y][Frame-1][Ch - 1];
				} // pull out data for this fit.
				fitParameters fitObj = new fitParameters(Coord, 
						testdata,
						Ch,
						Frame,
						inputPixelSize[Ch-1],
						gWindow[Ch-1],
						totalGain[Ch-1]);
				fitThese.add(fitObj);
			
			}
			int N = fitThese.size(); // number of particles to be fitted.
		    int[] gaussVector = new int [N*gWindow[Ch-1]*gWindow[Ch-1]];
		    float[] parameters = new float[N*7];
		    float[] bounds = {
					0.5F			, 1.5F,				// amplitude.
					1	, (gWindow[Ch-1]-1),			// x.
					1	, (gWindow[Ch-1]-1),			// y.
					0.7F			, (float) (gWindow[Ch-1] / 2.0),		// sigma x.
					0.7F			, (float) (gWindow[Ch-1] / 2.0),		// sigma y.
					(float) (-0.5*Math.PI) ,(float) (0.5*Math.PI),	// theta.
					-0.5F		, 0.5F				// offset.
			};
			
				
			float[] stepSize = new float[N*7];
		    for (int n = 0; n < N; n++) //loop over all parameters to set up calculations:
		    {
		    	for (int i = 0 ; i < gWindow[Ch-1]*gWindow[Ch-1]; i++){
		    		gaussVector[n*gWindow[Ch-1]*gWindow[Ch-1]+i] = fitThese.get(n).data[i]; // read in data
		    	}
		    	// start parameters for fit:
		    	parameters[n*7] = fitThese.get(n).data[gWindow[Ch-1]*(gWindow[Ch-1]-1)/2 + (gWindow[Ch-1]-1)/2]; // read in center pixel.
		    	parameters[n*7+1] = 2; // x center, will be calculated as weighted centroid on GPU.
		    	parameters[n*7+2] = 2; // y center, will be calculated as weighted centroid on GPU.
		    	parameters[n*7+3] = 1.6F; // x sigma.
		    	parameters[n*7+4] = 1.6F; // y sigma.
		    	parameters[n*7+5] = 0; // theta.
		    	parameters[n*7+6] = 0; // offset.
		    	stepSize[n*7] 	= 0.1F; // amplitude
		    	stepSize[n*7+1] 	= (float) (0.25*100/inputPixelSize[Ch-1]); // x
		    	stepSize[n*7+2] 	= (float) (0.25*100/inputPixelSize[Ch-1]); // y
		    	stepSize[n*7+3] 	= (float) (0.5*100/inputPixelSize[Ch-1]); // sigma x
		    	stepSize[n*7+4] 	= (float) (0.5*100/inputPixelSize[Ch-1]); // sigma y
		    	stepSize[n*7+5] 	= 0.1965F; //theta;
		    	stepSize[n*7+6] 	= 0.01F; // offset.
		    	
		    }
		    CUdeviceptr deviceGaussVector 	= CUDA.copyToDevice(gaussVector);
		    CUdeviceptr deviceParameters 	= CUDA.copyToDevice(parameters);
		    CUdeviceptr deviceStepSize 		= CUDA.copyToDevice(stepSize);
		    CUdeviceptr deviceBounds 		= CUDA.copyToDevice(bounds);
		    Pointer kernelParameters 		= Pointer.to(   
			    	Pointer.to(deviceGaussVector),
			        Pointer.to(new int[]{gaussVector.length}),
			        Pointer.to(deviceParameters),
			        Pointer.to(new int[]{parameters.length}),
			        Pointer.to(new int[]{gWindow[Ch-1]}),
			        Pointer.to(deviceBounds),
			        Pointer.to(new int[]{bounds.length}),
			        Pointer.to(deviceStepSize),
			        Pointer.to(new int[]{stepSize.length}),
			        Pointer.to(new double[]{convCriteria}),
			        Pointer.to(new int[]{maxIterations}));	
		    
		    int blockSizeX 	= 1;
		    int blockSizeY 	= 1;				   
		    int gridSizeX 	= (int) Math.ceil(Math.sqrt(N));
		    int gridSizeY 	= gridSizeX;
		    cuLaunchKernel(function,
		        gridSizeX,  gridSizeY, 1, 	// Grid dimension
		        blockSizeX, blockSizeY, 1,  // Block dimension
		        0, null,               		// Shared memory size and stream
		        kernelParameters, null 		// Kernel- and extra parameters
		    );
		    cuCtxSynchronize();
		    float hostOutput[] = new float[parameters.length];
		    // Pull data from device.
		    cuMemcpyDtoH(Pointer.to(hostOutput), deviceParameters,
		    		parameters.length * Sizeof.FLOAT);

		   // Free up memory allocation on device, housekeeping.
		    cuMemFree(deviceGaussVector);   
		    cuMemFree(deviceParameters);    
		    cuMemFree(deviceStepSize);
		    cuMemFree(deviceBounds);
		    for (int n = 0; n < N; n++) //loop over all particles
		    {	    	
				Particle Localized = new Particle();
				Localized.include 		= 1;
				Localized.channel 		= fitThese.get(n).channel;
				Localized.frame   		= fitThese.get(n).frame;
				Localized.r_square 		= hostOutput[n*7+6];
				Localized.x				= inputPixelSize[Ch-1]*(hostOutput[n*7+1] + fitThese.get(n).Center[0] - Math.round((gWindow[Ch-1])/2));
				Localized.y				= inputPixelSize[Ch-1]*(hostOutput[n*7+2] + fitThese.get(n).Center[1] - Math.round((gWindow[Ch-1])/2));
				Localized.z				= inputPixelSize[Ch-1]*0;	// no 3D information.
				Localized.sigma_x		= inputPixelSize[Ch-1]*hostOutput[n*7+3];
				Localized.sigma_y		= inputPixelSize[Ch-1]*hostOutput[n*7+4];
				Localized.sigma_z		= inputPixelSize[Ch-1]*0; // no 3D information.
				Localized.photons		= (int) (hostOutput[n*7]/fitThese.get(n).totalGain);
				Localized.precision_x 	= Localized.sigma_x/Math.sqrt(Localized.photons);
				Localized.precision_y 	= Localized.sigma_y/Math.sqrt(Localized.photons);
				Localized.precision_z 	= Localized.sigma_z/Math.sqrt(Localized.photons);
			//	Results.add(Localized);
				System.out.println(Localized.r_square);
		    }
		} // loop over all channels.
	 	
	/*    ArrayList<Particle> cleanResults = new ArrayList<Particle>();
		for (int i = 0; i < Results.size(); i++)
		{
			if (Results.get(i).sigma_x > 0 &&
					Results.get(i).sigma_y > 0 &&
					Results.get(i).precision_x > 0 &&
					Results.get(i).precision_y > 0 &&
					Results.get(i).photons > 0 && 
					Results.get(i).r_square > 0)
			cleanResults.add(Results.get(i));
				
		}
		//return cleanResults; // end parallel computation by returning results.*/
 // end GPU computing.

    
	
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
