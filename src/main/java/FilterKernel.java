import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadDataEx;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

/* Copyright 2017 Kristoffer Bernhem.
 * This file is part of SMLocalizer.
 *
 *  SMLocalizer is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SMLocalizer is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SMLocalizer.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *
 * @author kristoffer.bernhem@gmail.com
 */

public class FilterKernel {

	double[][] kernel;
	public FilterKernel (double sigma) // Gaussian filter.
	{
		// generate 500 pixel wide vector and sum over each 100.
		sigma *= 100;
		int gWidth = 500;
		double[] g = new double[gWidth];
		for (int x = 0; x < gWidth; x++)
			g[x] = 1/(sigma*Math.sqrt(2*Math.PI)) * Math.exp(-0.5*((x-(0.5*gWidth-0.5))/sigma)*((x-(0.5*gWidth-0.5))/sigma));
		double[] tempKernel = new double[gWidth/100];

		for (int i = 0; i < gWidth; i++)
		{
			tempKernel[i/100] += g[i];
		}			

		double[][] kernel = new double[gWidth/100][gWidth/100];
		double sum = 0 ;
		for (int x = 0; x < gWidth/100; x++)
		{
			for (int y = 0; y < gWidth/100; y++)
			{
				kernel[x][y] = tempKernel[x]*tempKernel[y];
				sum += tempKernel[x]*tempKernel[y];
			}
		}
		sum = 1/sum;
		for (int x = 0; x < gWidth/100; x++)
		{
			for (int y = 0; y < gWidth/100; y++)
			{
				kernel[x][y] *= sum;

			}
		}
		this.kernel = kernel;
	}
	public FilterKernel () // B-spline (bicubic) filter.
	{
		// generate 500 pixel wide vector and sum over each 100.
		int gWidth = 500;
		double[] g = new double[gWidth];
		double absX = 0;
		double alpha = 1.5; // set to 1 for bicubic.
		for (int x = 0; x < gWidth; x++) 
		{
			absX = Math.abs(x-(gWidth/2-0.5))/100;
			if (absX <= 1)
			{				
				g[x] = (alpha+2)*absX*absX*absX - (alpha+3)*absX*absX + 1;
				//	g[x] = 2/3 - absX*absX + (absX*absX*absX)/2;	
				//			g[x] = 2/3 - absX*absX + 0.5*absX*absX*absX;
			}else if (absX < 2)
			{		
				g[x] = alpha*absX*absX*absX - 5*alpha*absX*absX+8*alpha*absX-4*alpha;
				//	g[x] = (2-absX)*(2-absX)/6;
				//				g[x] = (2-absX)*(2-absX)*(2-absX)/6;
			}
			else if(absX >= 2)
				g[x] = 0;						

		}

		double[] tempKernel = new double[gWidth/100];

		for (int i = 0; i < gWidth; i++)
		{
			tempKernel[i/100] += g[i];

		}			

		double[][] kernel = new double[gWidth/100][gWidth/100];
		double sum = 0 ;
		for (int x = 0; x < gWidth/100; x++)
		{
			for (int y = 0; y < gWidth/100; y++)
			{
				kernel[x][y] = tempKernel[x]*tempKernel[y];
				sum += tempKernel[x]*tempKernel[y];
			}
		}
		sum = 1/sum;
		for (int x = 0; x < gWidth/100; x++)
		{
			for (int y = 0; y < gWidth/100; y++)
			{
				kernel[x][y] *= sum;

			}
		}
		this.kernel = kernel;
	}

	public int[][] filter(int[][] inpArray)
	{
		int[][] outArray = new int[inpArray.length][inpArray[0].length];
		
		for (int x = 0; x < inpArray.length; x++) 
		{
			for (int y = 0; y < inpArray[0].length; y++)
			{
				// apply 2D kernel (5x5)
				for (int i = -kernel.length/2; i <= kernel.length/2; i++)
				{
					if (x+i < inpArray.length && x+i >=0)
					{
						for (int j = -kernel.length/2; j <= kernel.length/2; j++)
						{
							if (y+j < inpArray[0].length && y+j >=0)
							{
								outArray[x][y] += inpArray[x+i][y+j]*kernel[i+2][j+2];							
								if (outArray[x][y] < 0)
									outArray[x][y] = 0;
							}
						}
					}
				}

			}						
		}

		return outArray;
	}

	public int[] filterGPU(int[] inpArray, int columns, int rows)
	{
		JCudaDriver.setExceptionsEnabled(true);
		// Initialize the driver and create a context for the first device.
		cuInit(0);
		CUdevice device = new CUdevice();
		cuDeviceGet(device, GlobalCUDAProps.CUDADeviceIndex);
		CUcontext context = new CUcontext();
		cuCtxCreate(context, 0, device);
		String ptxFileNameBspline = "filterImage.ptx";
		byte ptxFileBspline[] = CUDA.loadData(ptxFileNameBspline);
		CUmodule moduleBSpline = new CUmodule();
		CUfunction functionBpline = new CUfunction();

		cuModuleLoadDataEx(moduleBSpline, Pointer.to(ptxFileBspline), 
	            0, new int[0], Pointer.to(new int[0]));
		cuModuleGetFunction(functionBpline, moduleBSpline, "filterKernel");
		int[] output = new int[inpArray.length]; // result vector.
		

		double[] filterKernel = { 0.0015257568383789054, 0.003661718759765626, 0.02868598630371093, 0.0036617187597656254, 0.0015257568383789054, 
				0.003661718759765626, 0.008787890664062511, 0.06884453115234379, 0.00878789066406251, 0.003661718759765626, 
				0.02868598630371093, 0.06884453115234379, 0.5393295900878906, 0.06884453115234378, 0.02868598630371093,
				0.0036617187597656254, 0.00878789066406251, 0.06884453115234378, 0.008787890664062508, 0.0036617187597656254, 
				0.0015257568383789054, 0.003661718759765626, 0.02868598630371093, 0.0036617187597656254, 0.0015257568383789054};
		//	columns = 8;
		//	rows = 8;
		int bSplineDataLength = inpArray.length;
		CUdeviceptr device_Data_Bspline 		= CUDA.copyToDevice(inpArray); // median filtered data as input.
		CUdeviceptr deviceOutputBSpline 		= CUDA.allocateOnDevice(bSplineDataLength);
		CUdeviceptr deviceFilterKernel 			= CUDA.copyToDevice(filterKernel); // filter to applied to each pixel.				
		Pointer kernelParametersBspline 		= Pointer.to(   
				Pointer.to(device_Data_Bspline),	// input data.
				Pointer.to(new int[]{bSplineDataLength}),	// length of vector
				Pointer.to(new int[]{(columns)}), 	// width
				Pointer.to(new int[]{(rows)}),		// height
				Pointer.to(deviceFilterKernel),
				Pointer.to(new int[]{(int)filterKernel.length}),								
				Pointer.to(new int[]{(int)(Math.sqrt(filterKernel.length))}), // width of filter kernel.
				Pointer.to(deviceOutputBSpline),
				Pointer.to(new int[]{bSplineDataLength})
				);
		
		int blockSizeX 	= 1;
		int blockSizeY 	= 1;				   
		int gridSizeX 	= (int)Math.ceil(Math.sqrt(inpArray.length/(columns*rows)));
		int gridSizeY 	= gridSizeX;
		cuLaunchKernel(functionBpline,
				gridSizeX,  gridSizeY, 1, 	// Grid dimension
				blockSizeX, blockSizeY, 1,  // Block dimension
				0, null,               		// Shared memory size and stream
				kernelParametersBspline, null 		// Kernel- and extra parameters
				);
		cuCtxSynchronize();		
		cuMemcpyDtoH(Pointer.to(output), deviceOutputBSpline,
				bSplineDataLength * Sizeof.INT);
		
		cuMemFree(device_Data_Bspline);    
		cuMemFree(deviceOutputBSpline);    
		cuMemFree(deviceFilterKernel);    
		return output;
	}

	public static void main(String[] args) {
		long time = System.nanoTime();
		//	for (int i = 0; i < 10000; i++){
		//	gaussianSmoothing gs = new gaussianSmoothing(1.2);
		//FilterKernel gs = new FilterKernel();
		//}
		time = System.nanoTime()-time;
		time/= 1E6;
		System.out.println("time: " + time);
		FilterKernel gs = new FilterKernel();
		for(int i = 0; i < gs.kernel.length; i++)
		{
			for (int j = 0; j < gs.kernel[0].length; j++)
				System.out.print(gs.kernel[i][j] + " ");
			System.out.println("");
		}
		int[][] inpArray = new int[8][8];
		inpArray[2][2] = 1000;
	//	inpArray[3][1] = 500;
		for (int y = 0; y < 8; y++)
		{
			for (int x = 0; x < 8; x++)
			{
				System.out.print(inpArray[x][y] + ", ");
			}
			System.out.println("");
		}
		System.out.println("");
		inpArray = gs.filter(inpArray) ;
		for (int y = 0; y < 8; y++)
		{
			for (int x = 0; x < 8; x++)
			{
				System.out.print(inpArray[x][y] + " ");
			}
			System.out.println("");
		}

		
		
		int[] data = {	0, 0, 0, 0, 0, 0, 0, 0, 
				0, 0, 0, 0, 0, 0, 0, 0, 
				0, 0, 1000, 0, 0, 0, 0, 0, 
				0, 0, 0, 0, 0, 0, 0, 0, 
				0, 0, 0, 0, 0, 0, 0, 0, 
				0, 0, 0, 0, 0, 0, 0, 0, 
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, };
		int[] data2 = new int[data.length*1];
		int idx = 0;
		for (int i = 0; i < data2.length; i++)
		{
			data2[i] = data[idx];
			idx++;
			if (idx == data.length)
				idx=0;
		}
		int columns = 8;
		int rows = 8;
		int[] out =	gs.filterGPU(data2, columns, rows);
		System.out.println("**************" + "\n" + "*****************");
		for (int i = 0; i < data2.length; i++)
		{
			System.out.print(out[i] +" ");
			if (i % columns == columns - 1)
				System.out.println("");
			if (i%(columns*rows) == columns*rows-1)
				System.out.println("**********************");
		}
	}
}
