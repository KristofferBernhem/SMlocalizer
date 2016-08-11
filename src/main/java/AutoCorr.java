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

/*
 * v1.0 2016-07-18 Kristoffer Bernhem. Calculates correlation between two groups of Particles and 
 * by shifting the second group tries to maximize the correlation to approximate shift between the 
 * two groups.
 */
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import net.imagej.ImageJ;

public class AutoCorr {
	ArrayList<Particle> 
		referenceParticle,		// Reference, check the other list against this one.
		shiftParticle;			// Shift this one to maximize correlation between the two lists.
	int[] stepSize;				// x, y and z step size in shift calculations. 
	int[] maxShift;				// Maximal shift to calculate.

	AutoCorr(ArrayList<Particle> Alpha, ArrayList<Particle> Beta, int[] stepSize, int[] maxShift){
		this.referenceParticle 	= Alpha;
		this.shiftParticle 		= Beta;
		this.stepSize 			= stepSize;
		this.maxShift 			= maxShift;
	}

	public double Correlation(int[] shift){  // Calculate correlation for the current shift (x, y and z).
		double Corr = 0;

		int maxDistance = 500*250;  														// max distance in any 1 dimension.
		for(int referenceIndex = 0; referenceIndex < referenceParticle.size(); referenceIndex++){				// Loop over all referenceParticles.								
			for (int shiftIndex = 0; shiftIndex < shiftParticle.size(); shiftIndex++){			// For each referenceParticle, find the shiftParticles that are close.
				double xDist = (referenceParticle.get(referenceIndex).x - 					// Distance in x dimension after shift.
						shiftParticle.get(shiftIndex).x - 
						shift[0]);
				xDist *= xDist;
				if(xDist<maxDistance){												// If distance is small enough, check y.
					double yDist = (referenceParticle.get(referenceIndex).y - 				// Distance in y dimension after shift. 
							shiftParticle.get(shiftIndex).y - 
							shift[1]); 
					yDist *= yDist;
					if(yDist<maxDistance){											// If distance is small enough, check z.
						double zDist = (referenceParticle.get(referenceIndex).z -  			// Distance in z dimension after shift.
								shiftParticle.get(shiftIndex).z - 
								shift[2]);
						zDist *= zDist;
						if(zDist<maxDistance){										// If distance is small enough, calculate square distance.														
							double Distance = xDist+
									yDist+
									zDist;
							if (Distance == 0){										// Avoid assigning infinity as value.
								Corr += 1;
							}else{
								Corr += 1/Distance;									// Score of how close the particles were.
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
		int[] shift 					= {0,0,0};													// Output, initialize.
		int processors 					= Runtime.getRuntime().availableProcessors();				// Number of processor cores on this system.
		ExecutorService exec 			= Executors.newFixedThreadPool(processors);					// Set up parallel computing using all cores.
		List<Callable<double[]>> tasks 	= new ArrayList<Callable<double[]>>();						// Preallocate.
		
		
		/*
		 * Coarse round, find approximation of shift. Computation split this way to speed up performance. 
		 */
		
		for (int shiftX = - maxShift[0]; shiftX < maxShift[0]; shiftX += 3 * stepSize[0]){  		// Loop over all possible x coordinates in 3xstepsize steps
			for (int shiftY = - maxShift[1]; shiftY < maxShift[1]; shiftY += 3 *stepSize[1]){		// Loop over all possible y coordinates in 3xstepsize steps
				for (int shiftZ = - maxShift[2]; shiftZ < maxShift[2]; shiftZ += 3 * stepSize[2]){	// Loop over all possible z coordinates in 3xstepsize steps
					final int[] shiftEval = {shiftX,shiftY,shiftZ};									// Summarize evaluation parameters.
					Callable<double[]> c = new Callable<double[]>() {								// Computation to be done.
						@Override
						public double[] call() throws Exception {									
							double[] Correlation = {												// Results vector, containing correlation and the shifts generating that correlation score.
									Correlation(shiftEval),
									shiftEval[0],
									shiftEval[1],
									shiftEval[2]};						
							return Correlation;														// Actual call for each parallel process.
						}
					};
					tasks.add(c);																	// Que this task.
				}
			}
		}
		try {
			List<Future<double[]>> parallelCompute = exec.invokeAll(tasks);							// Execute computation.
			double[] Corr;
			double maxCorr = 0;

			for (int i = 1; i < parallelCompute.size(); i++){										// Loop over and transfer results.
				try {
					Corr = parallelCompute.get(i).get();
					if (Corr[0] > maxCorr){															// If a higher correlation score has been found.
						maxCorr = Corr[0];															// Update best value.
						shift[0] = (int) Corr[1];													// Update best guess at x shift.
						shift[1] = (int) Corr[2];													// Update best guess at y shift.
						shift[2] = (int) Corr[3];													// Update best guess at z shift.

					}
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	
		/*
		 * Second round, smaller stepsize this time round.
		 */
		
		System.out.println(shift[0]);
		List<Callable<double[]>> tasksSmall = new ArrayList<Callable<double[]>>();				// Preallocate.

		for (int shiftX = shift[0]-2 * stepSize[0]; shiftX < shift[0]+2 * stepSize[0]; shiftX += stepSize[0]){			// Loop over all x values surrounding the optimal shift value from coarse round.
			for (int shiftY = shift[1]-2 * stepSize[0]; shiftY <  shift[1]+2 * stepSize[1]; shiftY += stepSize[1]){		// Loop over all y values surrounding the optimal shift value from coarse round.	
				for (int shiftZ = shift[2]-2 * stepSize[0]; shiftZ <  shift[2]+2 * stepSize[2]; shiftZ += stepSize[2]){	// Loop over all z values surrounding the optimal shift value from coarse round.
					final int[] shiftEval = {shiftX,shiftY,shiftZ};														// Summarize evaluation parameters.
					Callable<double[]> c = new Callable<double[]>() {													// Computation to be done.
						@Override
						public double[] call() throws Exception {									
							double[] Correlation = {																	// Results vector, containing correlation and the shifts generating that correlation score.
									Correlation(shiftEval),
									shiftEval[0],
									shiftEval[1],
									shiftEval[2]};						
							return Correlation;																			// Actual call for each parallel process.
						}
					};
					tasksSmall.add(c);																					// Que this task.
				}
			}
		}
		
		try {
			List<Future<double[]>> parallelComputeSmall = exec.invokeAll(tasksSmall);		// Execute computation.
			double[] Corr;
			double maxCorr = 0;

			for (int i = 1; i < parallelComputeSmall.size(); i++){							// Loop over and transfer results.
				try {
					Corr = parallelComputeSmall.get(i).get();
					if (Corr[0] > maxCorr){													// If a higher correlation score has been found.
						maxCorr = Corr[0];													// Update best value.
						shift[0] = (int) Corr[1];											// Update best guess at x shift.
						shift[1] = (int) Corr[2];											// Update best guess at y shift.
						shift[2] = (int) Corr[3];											// Update best guess at z shift.
					}
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
		} catch (InterruptedException e) {

			e.printStackTrace();
		}
		finally {
			exec.shutdown();	// Shut down connection to cores.
		}	

		return shift; // Return optimal shift.
	} // Optimize.

	/* 
	 * function  tests.
	 */
	public static void main(String[] args){
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();
		Particle P = new Particle();
		P.x = 100;
		P.y = 100;
		P.z = 50;		
		ArrayList<Particle> A = new ArrayList<Particle>();
		ArrayList<Particle> B = new ArrayList<Particle>();
		double drift = 0.5;
		for (double i = 0; i < 200; i++){
			Particle P2 = new Particle();
			P2.x = P.x - i*drift;
			P2.y = P.y - i*drift;
			P2.z = P.z - i*drift;
			
			A.add(P2);
			Particle P3 = new Particle();
			P3.x = P.x + i*drift;
			P3.y = P.y + i*drift;
			P3.z = P.z + i*drift;
								
			if (i == 250)
			{
				System.out.println("A:" + P2.x);
				System.out.println("B:" + P3.x);
			}
			B.add(P3);
		}
		System.out.println(A.get(150).x);
		System.out.println(B.get(150).x);
		int[] stepSize = {5,5,5}; 			// shift will be rounded to these numbers.
		int[] maxShift = {250,250,250};		// maximal shift (+/-).
		long start = System.nanoTime();
		AutoCorr AC = new AutoCorr(A,B,stepSize,maxShift);

		int shift[] = AC.optimize();
		long stop = System.nanoTime();
		long elapsed = (stop-start)/1000000;
		System.out.println(shift[0]+  "x" +shift[1]+"x"+shift[2] + " in " + elapsed + " ms");
	}

}
