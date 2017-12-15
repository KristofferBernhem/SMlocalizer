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
import jcuda.driver.JCudaDriver;
import ome.xml.model.enums.EnumerationException;


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
		int pixelSizeZ = 10; // pixelsize for correlation images, will be improved upon once for final precision of 10 nm.
		int[] size = {(int)(width/pixelSize), (int)(height/pixelSize),1};


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
		size[2] = 1 + (int) (Math.ceil(zMax-zMin)/pixelSizeZ);
		for (int i = 0; i < 10; i++)
		{
			boundry[0][i] /= pixelSize;
			boundry[1][i] /= pixelSizeZ;
		}
		if (twoD)								
			size[2] = 1; // 2D data.
		try
		{
			if (selectedModel == 0)// parallel.
			{
				correctedResults = ImageCrossCorr3D.run(locatedParticles, nBins, boundry, size ,pixelSize,pixelSizeZ);
			}else
			{
				correctedResults = ImageCrossCorr3DGPU.run(locatedParticles, nBins, boundry, size ,pixelSize,pixelSizeZ);
			}
			TableIO.Store(correctedResults);
		}catch (IndexOutOfBoundsException e)
		{
			//ij.IJ.log("drift corrections failed");
		}

	}
	public static void ChannelAlign(int[][] boundry, int selectedModel){

		ArrayList<Particle> locatedParticles = TableIO.Load(); // Get current table data.		
		ArrayList<Particle> correctedResults = new ArrayList<Particle>(); // Output arraylist, will contain all particles that fit user input requirements after drift correction.
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();
		double width = tab.getValue("width", 0);
		double height = tab.getValue("height", 0);
		int pixelSize = 20; // pixelsize for correlation images, will be improved upon once for final precision of 10 nm.
		int pixelSizeZ = 40; // pixelsize for correlation images, will be improved upon once for final precision of 10 nm.
		int[] size = {1+(int)(width/pixelSize), 1+(int)(height/pixelSize),1};
		if (locatedParticles.size() == 0)
		{
			ij.IJ.log("No data to align.");
			return;
		}


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
		size[2] =1 + (int) (Math.ceil(zMax-zMin)/pixelSizeZ);
		for (int i = 0; i < 10; i++)
		{
			boundry[0][i] /= pixelSize;
			boundry[1][i] /= pixelSizeZ;
		}
		if (twoD)

			size[2] = 1; // 2D data.
		try
		{
			if (selectedModel == 0)// parallel.
			{
				correctedResults = ImageCrossCorr3D.runChannel(locatedParticles, boundry, size ,pixelSize,pixelSizeZ);
			}else
			{
				correctedResults = ImageCrossCorr3DGPU.runChannel(locatedParticles, boundry, size ,pixelSize,pixelSizeZ);
			}

			TableIO.Store(correctedResults);
		}catch (IndexOutOfBoundsException e)
		{
			ij.IJ.log("channel alignment failed");
		}
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
	static void plot(double[] values, double[] values2, double[] values3, double[] x_axis, int Ch) {
		Plot newPlot = new Plot("Drift corrections channel: " + Ch,"frame","drift [nm]");
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
