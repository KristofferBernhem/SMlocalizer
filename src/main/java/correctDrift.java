/* Copyright 2016 Kristoffer Bernhem.
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
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadDataEx;

import java.awt.Color;
import java.util.ArrayList;

import ij.gui.Plot;
import ij.plugin.filter.Analyzer;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;


public class correctDrift {

	/*
	 * test function.
	 */
	public static void main(String[] args){ // test case.
		Particle P = new Particle();
		P.x = 100;
		P.y = 100;
		//P.z = 50;		
		P.channel = 1;
		Particle Psecond = new Particle();
		Psecond.x = 1000;
		Psecond.y = 1000;
		//Psecond.z = 500;
		Psecond.channel = 1;
		ArrayList<Particle> A = new ArrayList<Particle>();
		double drift = 0.20;
		for (double i = 1; i < 2000; i++){
			Particle P2 = new Particle();
			P2.x = P.x - i*drift;
			P2.y = P.y - i*drift;
		//	P2.z = P.z - 2*i*drift;
			P2.channel = 1;
			P2.include = 1;
			P2.frame = (int) i;

			A.add(P2);

			Particle P4 = new Particle();
			P4.x = Psecond.x - i*drift;
			P4.y = Psecond.y - i*drift;
	//		P4.z = Psecond.z - 2*i*drift;
			P4.frame = (int) i;
			P4.channel = 1;
			P4.include = 1;
			A.add(P4);

		}
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);
		TableIO.Store(A);

		int[][] boundry = new int[2][10];
		int[] nBins = new int[10];
		int[] nParticles = new int[10];
		int[] minParticles = new int[10];
		for (int i = 0; i < 10;i++)
		{
			boundry[0][i] = 250;
			boundry[1][i] = 250;

			nBins[i] = 10;
			nParticles[i] = 1000;
			minParticles[i] = 10;					

		}
		run(boundry,nBins,0);
	}

	/*
	 * drift correct the fitted results table.
	 */
	public static void run(int[][] boundry, int[] nBins, int selectedModel){
		//int[] maxDistance = {2500,2500,2500}; // everything beyond 50 nm apart after shift will not affect outcome.
		ArrayList<Particle> locatedParticles = TableIO.Load(); // Get current table data.		
		ArrayList<Particle> correctedResults = new ArrayList<Particle>(); // Output arraylist, will contain all particles that fit user input requirements after drift correction.
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();

		if (locatedParticles.size() == 0)
		{
			ij.IJ.log("No data to align.");
			return;
		}
		double width = tab.getValue("width", 0);
		double height = tab.getValue("height", 0);
		int pixelSize = 10; // pixelsize for correlation images, will be improved upon once for final precision of 10 nm.
		int pixelSizeZ = 20; // pixelsize for correlation images, will be improved upon once for final precision of 10 nm.
		int[] size = {(int)(width/pixelSize), (int)(height/pixelSize),1};

		if (selectedModel == 0)// parallel.
		{
		
			boolean twoD = true;
			int idx = 0;
			double zMin = 0;
			double zMax = 0;
			while (idx < locatedParticles.size())
			{
				if (locatedParticles.get(idx).z != 0)
				{
					twoD = false;
					if (locatedParticles.get(idx).z < zMin)
						zMin = locatedParticles.get(idx).z ;
					else if (locatedParticles.get(idx).z > zMax)
						zMax = locatedParticles.get(idx).z ;
				}
				idx++;
			}
			size[2] = (int) (Math.ceil(zMax-zMin));
			for (int i = 0; i < 10; i++)
			{
				boundry[0][i] /= pixelSize;
				boundry[1][i] /= pixelSizeZ;
			}
			if (twoD)
			{					
				size[2] = 1; // 2D data.
				correctedResults = ImageCrossCorr3D.run(locatedParticles, nBins, boundry, size ,pixelSize,pixelSizeZ,true);
			}
			else
				correctedResults = ImageCrossCorr3D.run(locatedParticles, nBins, boundry, size ,pixelSize,pixelSizeZ,true);
			
			
			TableIO.Store(correctedResults);
		}else // GPU.
		{
			//TODO add code.
		}//end gpu
		

	}
	public static void ChannelAlign(int[][] boundry, int selectedModel){
	
		ArrayList<Particle> locatedParticles = TableIO.Load(); // Get current table data.		
		ArrayList<Particle> correctedResults = new ArrayList<Particle>(); // Output arraylist, will contain all particles that fit user input requirements after drift correction.
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();
		double width = tab.getValue("width", 0);
		double height = tab.getValue("height", 0);
		int pixelSize = 10; // pixelsize for correlation images, will be improved upon once for final precision of 10 nm.
		int pixelSizeZ = 20; // pixelsize for correlation images, will be improved upon once for final precision of 10 nm.
		int[] size = {(int)(width/pixelSize), (int)(height/pixelSize),1};
		if (locatedParticles.size() == 0)
		{
			ij.IJ.log("No data to align.");
			return;
		}

		if (selectedModel == 0)// parallel.
		{
			boolean twoD = true;
			int idx = 0;
			double zMin = 0;
			double zMax = 0;
			while (idx < locatedParticles.size())
			{
				if (locatedParticles.get(idx).z != 0)
				{
					twoD = false;
					if (locatedParticles.get(idx).z < zMin)
						zMin = locatedParticles.get(idx).z ;
					else if (locatedParticles.get(idx).z > zMax)
						zMax = locatedParticles.get(idx).z ;
				}
				idx++;
			}
			size[2] = (int) (Math.ceil(zMax-zMin));
			for (int i = 0; i < 10; i++)
			{
				boundry[0][i] /= pixelSize;
				boundry[1][i] /= pixelSizeZ;
			}
			if (twoD)
			{
				size[2] = 1; // 2D data.
				correctedResults = ImageCrossCorr3D.runChannel(locatedParticles, boundry, size ,pixelSize,pixelSizeZ,true);
			}
			else
				correctedResults = ImageCrossCorr3D.runChannel(locatedParticles, boundry, size ,pixelSize,pixelSizeZ,true);
			
			
			TableIO.Store(correctedResults);
		}else // GPU
		{
			
		} // end gpu.
	/*
			}else // end parallel-
				if (selectedModel == 2) // GPU
				{
					// Initialize the driver and create a context for the first device.
					cuInit(0);
					CUdevice device = new CUdevice();
					cuDeviceGet(device, 0);
					CUcontext context = new CUcontext();
					cuCtxCreate(context, 0, device);
					// Load the PTX that contains the kernel.					
					CUmodule module = new CUmodule();
					String ptxFileName = "driftCorr.ptx";
					byte ptxFile[] = CUDA.loadData(ptxFileName);				
					cuModuleLoadDataEx(module, Pointer.to(ptxFile), 
				            0, new int[0], Pointer.to(new int[0]));
//					cuModuleLoad(module, "driftCorr.ptx");
					// Obtain a handle to the kernel function.
					CUfunction function = new CUfunction();
					cuModuleGetFunction(function, module, "run");
					// setup retained kernel parameters:
					int[] stepSize = {5,5}; // [x/y,z].
					CUdeviceptr device_stepSize 		= CUDA.copyToDevice(stepSize); // stepsize in xy and z.


					for (int Ch = 2; Ch <= Channels; Ch++)
					{

						int[] maxShift 				= {boundry[0][Ch-1],  //shift in xy.
								boundry[1][Ch-1]}; // shift in z.
						int[] numStep 				= {(int) (2*Math.round(maxShift[0]/stepSize[0]) + 1), // number of steps in xy.
								1}; 			// number of steps in z, will be updated in the code below.
						CUdeviceptr device_maxShift = CUDA.copyToDevice(maxShift);



						ArrayList<Particle> Data1 	= new ArrayList<Particle>(); 		// Target particles.			
						int addedFrames1 			= 0;								// Number of particles added to the bin.
						int index = 0;
						int z = 0;
						while (addedFrames1 < nParticles[Ch-2] && index < locatedParticles.size())
						{
							if (locatedParticles.get(index).channel == Ch-1 &&
									locatedParticles.get(index).include == 1){
								Data1.add(locatedParticles.get(index));					
								addedFrames1++;
								z+=(int)locatedParticles.get(index).z;
							}
							index++;
						} // load Data 1.
						ArrayList<Particle> Data2 	= new ArrayList<Particle>(); 		// Change these particles so that the correlation function is maximized.
						int addedFrames2 			= 0;								// Number of particles added to the bin.
						index = 0;
						while (addedFrames2 < nParticles[Ch-1] && index < locatedParticles.size()){
							if (locatedParticles.get(index).channel == Ch &&
									locatedParticles.get(index).include == 1){
								Data2.add(locatedParticles.get(index));					
								addedFrames2++;
								z+=(int)locatedParticles.get(index).z;
							}
							index++;
						} // Load Data 2.

						ArrayList<Particle> Beta = hasNeighbors(Data1, Data2, (double) maxDistance[0]);
						ArrayList<Particle> Alpha = hasNeighbors(Beta, Data1, (double) maxDistance[0]);
						if(Alpha.size() < minParticles[Ch-2])
						{
							ij.IJ.log("not enough particles, no alignment possible");
							return;
						}
						if(Beta.size() < minParticles[Ch-1])
						{
							ij.IJ.log("not enough particles, no alignment possible");
							return;
						}
						int[] lambdaCh = {0,0,0,0}; // initiate.


						/*
						 * Load data to device:
						 */
		/*				if (z == 0) // 2D samples.
						{
							int[] referenceParticles = new int[2*Alpha.size()];
							int[] targetParticles 	 = new int[2*Beta.size()];
							for (int i = 0; i < Alpha.size(); i++)
							{
								referenceParticles[i*2]   = (int)Alpha.get(i).x; 
								referenceParticles[i*2+1] = (int)Alpha.get(i).y; 									
							}
							for (int i = 0; i < Beta.size(); i++)
							{
								targetParticles[i*2]   = (int)Beta.get(i).x; 
								targetParticles[i*2+1] = (int)Beta.get(i).y; 									
							} // 2D ends
							numStep[1] = 1;
							CUdeviceptr device_numStep 				= CUDA.copyToDevice(numStep);
							CUdeviceptr device_referenceParticles 	= CUDA.copyToDevice(referenceParticles);
							CUdeviceptr device_targetParticles 		= CUDA.copyToDevice(targetParticles);
							CUdeviceptr device_result 				= CUDA.allocateOnDevice((int)(numStep[0]*numStep[0])); // swap vector.
							int N 									= (int)Math.ceil(Math.sqrt(numStep[0]*numStep[0]));
							Pointer kernelParameters 	= Pointer.to(   
									Pointer.to(device_referenceParticles),
									Pointer.to(new int[]{referenceParticles.length}),
									Pointer.to(device_targetParticles),
									Pointer.to(new int[]{targetParticles.length}),
									Pointer.to(device_maxShift),
									Pointer.to(new int[]{2}),
									Pointer.to(device_stepSize),
									Pointer.to(new int[]{2}),
									Pointer.to(device_numStep),
									Pointer.to(new int[]{2}),
									Pointer.to(device_result),
									Pointer.to(new int[]{numStep[0]*numStep[0]})
									);
							int blockSizeX 	= 1;
							int blockSizeY 	= 1;				   
							int gridSizeX 	= N;
							int gridSizeY 	= N;
							cuLaunchKernel(function,
									gridSizeX,  gridSizeY, 1, 	// Grid dimension
									blockSizeX, blockSizeY, 1,  // Block dimension
									0, null,               		// Shared memory size and stream
									kernelParameters, null 		// Kernel- and extra parameters
									);
							cuCtxSynchronize();

							// Pull data from device.
							int hostOutput[] = new int[numStep[0]*numStep[0]];
							cuMemcpyDtoH(Pointer.to(hostOutput), device_result,
									hostOutput.length * Sizeof.INT);

							// Free up memory allocation on device, housekeeping.
							cuMemFree(device_referenceParticles);   
							cuMemFree(device_targetParticles);    
							cuMemFree(device_result);
							cuMemFree(device_numStep); 
							cuMemFree(device_stepSize); 
							cuMemFree(device_maxShift); 

							// return data.
							int corr = 0;
							for (int i = 0; i < hostOutput.length; i++)
							{
								if (hostOutput[i] > corr)
								{
									corr = hostOutput[i];
									lambdaCh[1] = maxShift[0] - (i / (numStep[0] * numStep[1])) * stepSize[0];
									lambdaCh[2] = maxShift[0] - (i % numStep[0]) * stepSize[0];									
								}
							} // get best estimate of drift.
							lambdaCh[3] = 0;
						}else // 3D sample.
						{
							int[] referenceParticles = new int[3*Alpha.size()];
							int[] targetParticles = new int[3*Beta.size()];
							for (int i = 0; i < Alpha.size(); i++)
							{
								referenceParticles[i*3] = (int)Alpha.get(i).x; 
								referenceParticles[i*3+1] = (int)Alpha.get(i).y; 
								referenceParticles[i*3+2] = (int)Alpha.get(i).z; 
							}
							for (int i = 0; i < Beta.size(); i++)
							{
								targetParticles[i*3]   = (int)Beta.get(i).x; 
								targetParticles[i*3+1] = (int)Beta.get(i).y; 
								targetParticles[i*3+2] = (int)Beta.get(i).z; 
							}
							numStep[1] 								= (int) (2*Math.round(maxShift[1]/stepSize[1]) + 1); 			// number of steps in z.
							CUdeviceptr device_numStep 				= CUDA.copyToDevice(numStep);
							CUdeviceptr device_referenceParticles 	= CUDA.copyToDevice(referenceParticles);
							CUdeviceptr device_targetParticles 		= CUDA.copyToDevice(targetParticles);
							CUdeviceptr device_result 				= CUDA.allocateOnDevice((int)(numStep[0]*numStep[0]*numStep[1])); // swap vector.
							int N = (int)Math.ceil(Math.sqrt(numStep[0]*numStep[0]*numStep[1]));
							Pointer kernelParameters 	= Pointer.to(   
									Pointer.to(device_referenceParticles),
									Pointer.to(new int[]{referenceParticles.length}),
									Pointer.to(device_targetParticles),
									Pointer.to(new int[]{targetParticles.length}),
									Pointer.to(device_maxShift),
									Pointer.to(new int[]{2}),
									Pointer.to(device_stepSize),
									Pointer.to(new int[]{2}), 
									Pointer.to(device_numStep),
									Pointer.to(new int[]{2}),
									Pointer.to(device_result),
									Pointer.to(new int[]{numStep[0]*numStep[0]*numStep[1]})
									);
							int blockSizeX 	= 1;
							int blockSizeY 	= 1;				   
							int gridSizeX 	= N;
							int gridSizeY 	= N;
							cuLaunchKernel(function,
									gridSizeX,  gridSizeY, 1, 	// Grid dimension
									blockSizeX, blockSizeY, 1,  // Block dimension
									0, null,               		// Shared memory size and stream
									kernelParameters, null 		// Kernel- and extra parameters
									);
							cuCtxSynchronize();

							// Pull data from device.
							int hostOutput[] = new int[numStep[0]*numStep[0]*numStep[1]];
							cuMemcpyDtoH(Pointer.to(hostOutput), device_result,
									hostOutput.length * Sizeof.INT);

							// Free up memory allocation on device, housekeeping.
							cuMemFree(device_referenceParticles);   
							cuMemFree(device_targetParticles);    
							cuMemFree(device_result);
							cuMemFree(device_numStep);   
							cuMemFree(device_stepSize); 
							cuMemFree(device_maxShift); 
							int corr = 0;
							for (int i = 0; i < hostOutput.length; i++)
							{
								if (hostOutput[i] > corr)
								{
									corr = hostOutput[i];
									lambdaCh[1] = maxShift[0] - (i / (numStep[0] * numStep[1])) * stepSize[0];
									lambdaCh[2] = maxShift[0] - (i / numStep[1]) * stepSize[0];
									lambdaCh[3] = maxShift[1] - (i % numStep[1]) * stepSize[1];
								}
							} // get best estimate of drift.
						} // 3D ends.


						ij.IJ.log("Channel " + Ch + " shifted by " + lambdaCh[1]+  " x " + lambdaCh[2] + " x " + lambdaCh[3] + " nm.");

						for(int i = 0; i < locatedParticles.size(); i++)
						{
							if (locatedParticles.get(i).channel == Ch)
							{
								locatedParticles.get(i).x = locatedParticles.get(i).x + lambdaCh[1];
								locatedParticles.get(i).y = locatedParticles.get(i).y + lambdaCh[2];
								locatedParticles.get(i).z = locatedParticles.get(i).z + lambdaCh[3];
							}
						}		

						for (int i = locatedParticles.size()-1; i >=0; i--)
						{
							if(locatedParticles.get(i).x < 0 ||
									locatedParticles.get(i).y < 0 ||
									locatedParticles.get(i).z < 0)
							{
								locatedParticles.remove(i);
							}		
						} // verify that the particles have not been shifted out of bounds.			
					} // channel loop.
					
					TableIO.Store(locatedParticles);
					ij.IJ.log("Channels aligned.");					
				} // end GPU.*/
	}

	public static double[] interp(double X1, double X2, int n){
		double[] extendedX 	= new double[n]; 
		extendedX[0] 		= X1;
		extendedX[n-1] 		= X2;

		double step 		= (X2-X1)/(n-2);
		for (int i = 1; i < n-1; i++){
			extendedX[i] = extendedX[i-1] + step;
		}

		return extendedX;
	}

	/*
	 * Supporting plot functions
	 */
	static void plot(double[] values) {
		double[] x = new double[values.length];
		for (int i = 0; i < x.length; i++)
			x[i] = i;
		Plot plot = new Plot("Plot window", "x", "values", x, values);
		plot.show();
	}
	static void plot(double[] values,int[] x_axis) {
		double[] x = new double[values.length];
		for (int i = 0; i < x.length; i++)
			x[i] = x_axis[i];
		Plot plot = new Plot("Plot window", "x", "values", x, values);
		plot.show();
	}
	static void plot(double[] values, double[] values2,double[] x_axis) {
		Plot newPlot = new Plot("Drift corrections","frame","drift [nm]");
		newPlot.setColor(Color.GREEN);
		newPlot.addPoints(x_axis,values, Plot.LINE);
		newPlot.setColor(Color.RED);
		newPlot.addPoints(x_axis,values2, Plot.LINE);
		newPlot.addLegend("X \n Y");
		newPlot.show();		
	}
	static void plot(double[] values, double[] values2, double[] values3, double[] x_axis) {
		Plot newPlot = new Plot("Drift corrections","frame","drift [nm]");
		newPlot.setColor(Color.GREEN);
		newPlot.addPoints(x_axis,values, Plot.LINE);
		newPlot.setColor(Color.RED);
		newPlot.addPoints(x_axis,values2, Plot.LINE);
		newPlot.setColor(Color.BLUE);
		newPlot.addPoints(x_axis,values3, Plot.LINE);
		newPlot.addLegend("X \n Y \n Z");
		newPlot.show();		
	}
	public static ArrayList<Particle> hasNeighbors(ArrayList<Particle> Alpha, ArrayList<Particle> Beta, double maxDistance)
	{	
		ArrayList<Particle> Include = new ArrayList<Particle>();		
		boolean[] retainBeta = new boolean[Beta.size()];
		for (int i = 0; i < Alpha.size(); i++) // loop over all entries in Alpha.
		{
			double x = Alpha.get(i).x;
			double y = Alpha.get(i).y;
			double z = Alpha.get(i).z;

			for (int j = 0; j < Beta.size(); j++)
			{
				double distance = Math.sqrt(
						(x-Beta.get(j).x)*(x-Beta.get(j).x) +
						(y-Beta.get(j).y)*(y-Beta.get(j).y)+
						(z-Beta.get(j).z)*(z-Beta.get(j).z) );
				if (distance < maxDistance)
					retainBeta[j] = true;												
			}						
		}
		for (int i = 0; i < Beta.size(); i++)
		{
			if(retainBeta[i])
				Include.add(Beta.get(i));
		}

		return Include;
	}
}
