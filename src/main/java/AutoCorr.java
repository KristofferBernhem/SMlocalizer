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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class AutoCorr {
	ArrayList<Particle> Alpha,Beta;
	int[] stepSize;
	int[] maxShift;

	AutoCorr(ArrayList<Particle> Alpha, ArrayList<Particle> Beta, int[] stepSize, int[] maxShift){
		this.Alpha = Alpha;
		this.Beta = Beta;
		this.stepSize = stepSize;
		this.maxShift = maxShift;
	}

	public double Correlation(int[] shift){
		double Corr = 0;

		int maxDistance = 25*25;  // max distance in any 1 dimension.
		for(int index = 0; index < Alpha.size(); index++){			
			for (int index2 = 0; index2 < Beta.size(); index2++){
				double xDist = (Alpha.get(index).x - Beta.get(index2).x - shift[0]);
				xDist *= xDist;
				if(xDist<maxDistance){
					double yDist = (Alpha.get(index).y - Beta.get(index2).y - shift[1]); 
					yDist *= yDist;
					if(yDist<maxDistance){
						double zDist = (Alpha.get(index).z - Beta.get(index2).z - shift[2]);
						zDist *= zDist;
						if(zDist<maxDistance){						
							double Distance = xDist+
									yDist+
									zDist;
							if (Distance == 0){
								Corr += 1.5;
							}else{
								Corr += 1/Distance;								
							}							
						}	
					}
				}
			}
		}
		return Corr;
	}
/*
 * optimize takes the global settings and loops over and finds optimal shift between the two groups of Particles. 
 * 2 rounds of optimization, once at 3x stepsize for coarse optimization and once at stepsize for fine optimization. 
 * Runs on all cores and scales linearly with number of cores for computation time.
 */
	public int[] optimize(){
		int[] shift 					= {0,0,0};										// Output.
		int processors 					= Runtime.getRuntime().availableProcessors();	// Number of processor cores on this system.
		ExecutorService exec 			= Executors.newFixedThreadPool(processors);		// Set up parallel computing using all cores.
		List<Callable<double[]>> tasks 	= new ArrayList<Callable<double[]>>();			// Preallocate.

		for (int shiftX = - maxShift[0]; shiftX < maxShift[0]; shiftX += 3 * stepSize[0]){
			for (int shiftY = - maxShift[1]; shiftY < maxShift[1]; shiftY += 3 *stepSize[1]){
				for (int shiftZ = - maxShift[2]; shiftZ < maxShift[2]; shiftZ += 3 * stepSize[2]){
					final int[] shiftEval = {shiftX,shiftY,shiftZ};				
					Callable<double[]> c = new Callable<double[]>() {					// Computation to be done.
						@Override
						public double[] call() throws Exception {						
							double[] Correlation = {
									Correlation(shiftEval),
									shiftEval[0],
									shiftEval[1],
									shiftEval[2]};						
							return Correlation;						// Actual call for each parallel process.
						}
					};
					tasks.add(c);														// Que this task.
				}
			}

		}
		try {
			List<Future<double[]>> parallelCompute = exec.invokeAll(tasks);				// Execute computation.
			double[] Corr;
			double maxCorr = 0;

			for (int i = 1; i < parallelCompute.size(); i++){							// Loop over and transfer results.
				try {
					Corr = parallelCompute.get(i).get();
					if (Corr[0] > maxCorr){
						maxCorr = Corr[0];
						shift[0] = (int) Corr[1];
						shift[1] = (int) Corr[2];
						shift[2] = (int) Corr[3];

					}
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			//results.get(1).get().channel
		} catch (InterruptedException e) {

			e.printStackTrace();
		}

		List<Callable<double[]>> tasksSmall 	= new ArrayList<Callable<double[]>>();			// Preallocate.

		for (int shiftX = shift[0]-3 * stepSize[0]; shiftX < shift[0]+3 * stepSize[0]; shiftX += stepSize[0]){
			for (int shiftY = shift[1]-3 * stepSize[0]; shiftY <  shift[1]+3 * stepSize[1]; shiftY += stepSize[1]){
				for (int shiftZ = shift[2]-3 * stepSize[0]; shiftZ <  shift[2]+3 * stepSize[2]; shiftZ += stepSize[2]){
					final int[] shiftEval = {shiftX,shiftY,shiftZ};				
					Callable<double[]> c = new Callable<double[]>() {					// Computation to be done.
						@Override
						public double[] call() throws Exception {						
							double[] Correlation = {
									Correlation(shiftEval),
									shiftEval[0],
									shiftEval[1],
									shiftEval[2]};						
							return Correlation;						// Actual call for each parallel process.
						}
					};
					tasksSmall.add(c);														// Que this task.
				}
			}

		}
		try {
			List<Future<double[]>> parallelComputeSmall = exec.invokeAll(tasksSmall);				// Execute computation.
			double[] Corr;
			double maxCorr = 0;

			for (int i = 1; i < parallelComputeSmall.size(); i++){							// Loop over and transfer results.
				try {
					Corr = parallelComputeSmall.get(i).get();
					if (Corr[0] > maxCorr){
						maxCorr = Corr[0];
						shift[0] = (int) Corr[1];
						shift[1] = (int) Corr[2];
						shift[2] = (int) Corr[3];

					}
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			//results.get(1).get().channel
		} catch (InterruptedException e) {

			e.printStackTrace();
		}
		finally {
			exec.shutdown();
		}	

		return shift; // Return optimal shift.
	}

	/* 
	 * function  tests.
	 */
	public static void main(String[] args){
		Particle P = new Particle();
		P.x = 100;
		P.y = 100;
		P.z = 50;
		Particle P2 = new Particle();
		P2.x = 120;
		P2.y = 115;
		P2.z = 42;
		ArrayList<Particle> A = new ArrayList<Particle>();
		ArrayList<Particle> B = new ArrayList<Particle>();
		for (int i = 0; i < 500; i++){
			A.add(P);
			B.add(P2);
		}


		int[] stepSize = {5,5,5}; // shift will be rounded to these numbers.
		int[] maxShift = {250,250,250};
		long start = System.nanoTime();
		AutoCorr AC = new AutoCorr(A,B,stepSize,maxShift);

		int shift[] = AC.optimize();
		long stop = System.nanoTime();
		long elapsed = (stop-start)/1000000;
		System.out.println(shift[0]+  "x" +shift[1]+"x"+shift[2] + " in " + elapsed + " ms");
	}

}
