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

import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;


/*
 * Shift-Accumulation 2D autocorrelation.
 */
/* TODO
 * Test effects of varied gaussian blur on results.
 * Copy and create 3D version.
 */
public class autoCorr2D {
	int[][] imageAlpha, imageBeta;
	int width, height;

	autoCorr2D(int[][] imageAlpha, int[][] imageBeta, int width, int height){
		this.imageAlpha = imageAlpha;
		this.imageBeta = imageBeta;
		this.width = width;
		this.height = height;
	}

	autoCorr2D(ArrayList<Particle> imageAlpha, ArrayList<Particle> imageBeta, int width, int height){
		ShortProcessor IP_alpha = new ShortProcessor(width,height);
		IP_alpha.set(0); // Set to 0.
		for (int i = 0; i< imageAlpha.size(); i++){
			IP_alpha.set(
					(int)Math.round(imageAlpha.get(i).x), 
					(int)Math.round(imageAlpha.get(i).y), 
					10000);
		}
		ShortProcessor IP_beta = new ShortProcessor(width,height);
		IP_beta.set(0); // Set to 0.
		for (int i = 0; i< imageBeta.size(); i++){
			IP_beta.set(
					(int)Math.round(imageBeta.get(i).x), 
					(int)Math.round(imageBeta.get(i).y), 
					10000);
		}

		IP_alpha.blurGaussian(3); 	// 3 pixel search radius.
		IP_beta.blurGaussian(3); 	// 3 pixel search radius.

		this.imageAlpha = IP_alpha.getIntArray();
		this.imageBeta =  IP_beta.getIntArray();
		this.width = width;
		this.height = height;
	}
	public double Correlation(int[] shift){	
		/*
		 * Limit loop as 0 padding will result in the same value as exclusion.
		 */
		int minX = 0;
		int maxX = width;
		int minY = 0;
		int maxY = height;
		if (shift[0] < 0){
			maxX = width + shift[0];
		}else if(shift[0] > 0){
			minX = shift[0];
		}
		if (shift[1] < 0){
			maxY = height + shift[1];
		}else if(shift[1] > 0){
			minY = shift[1];
		}

		double Corr = 0;
		for(int i = minX; i < maxX; i ++){
			for (int j = minY; j < maxY; j++){
				Corr += imageAlpha[i][j]*imageBeta[i-shift[0]][j-shift[1]];
			}

		}
		return Corr;
	}

	public int[] optimize(int[] maxShift){
		int CoarseStep = 5;
		int[] shift = new int[2];
		double[][] Correlation = new double[maxShift[0]*2*2*maxShift[1]][3];
		int count = 0;
		for(int shiftX = -maxShift[0]; shiftX < maxShift[0]; shiftX += CoarseStep){
			for(int shiftY = -maxShift[1]; shiftY < maxShift[1]; shiftY += CoarseStep){
				int[] shiftEval = {shiftX,shiftY};
				Correlation[count][0] = Correlation(shiftEval);
				Correlation[count][1] = shiftX;
				Correlation[count][2] = shiftY;
				count++;
			}			
		}

		// done in this way for parallelization implementation later on.
		double minCorr = Correlation[0][0];
		shift[0] = (int) Correlation[0][1];
		shift[1] = (int) Correlation[0][2];
		for (int index = 1; index < Correlation.length; index++){
			if(Correlation[index][0] > minCorr){
				minCorr = Correlation[index][0];
				shift[0] = (int) Correlation[index][1];
				shift[1] = (int) Correlation[index][2];

			}
		}
		double[][] CorrelationSmaller = new double[((CoarseStep+1)*2+1)*((CoarseStep+1)*2+1)][3];
		count = 0;
		for(int shiftX = shift[0]-CoarseStep-1; shiftX < shift[0]+CoarseStep+1; shiftX ++){
			for(int shiftY = shift[1]-CoarseStep-1; shiftY < shift[1]+CoarseStep+1; shiftY ++){
				int[] shiftEval = {shiftX,shiftY};
				CorrelationSmaller[count][0] = Correlation(shiftEval);
				CorrelationSmaller[count][1] = shiftX;
				CorrelationSmaller[count][2] = shiftY;
				count++;
			}			
		}

		// done in this way for parallelization implementation later on.
		for (int index = 1; index < CorrelationSmaller.length; index++){
			if(CorrelationSmaller[index][0] > minCorr){
				minCorr = CorrelationSmaller[index][0];
				shift[0] = (int) CorrelationSmaller[index][1];
				shift[1] = (int) CorrelationSmaller[index][2];

			}
		}
		return shift;
	}

	public int[] optimizeParallel(int[] maxShift){
		int CoarseStep = 5;
		int[] shift = new int[2];
		List<Callable<double[]>> tasks = new ArrayList<Callable<double[]>>();	// Preallocate.
		for(int shiftX = -maxShift[0]; shiftX < maxShift[0]; shiftX += CoarseStep){
			for(int shiftY = -maxShift[1]; shiftY < maxShift[1]; shiftY += CoarseStep){
				final int[] shiftEval = {shiftX,shiftY};
				Callable<double[]> c = new Callable<double[]>() {					// Computation to be done.
					@Override
					public double[] call() throws Exception {						
						double[] Correlation = {
								Correlation(shiftEval),
								shiftEval[0],
								shiftEval[1]};						
						return Correlation;						// Actual call for each parallel process.
					}
				};
				tasks.add(c);														// Que this task.
			} 
		}

		int processors 			= Runtime.getRuntime().availableProcessors();	// Number of processor cores on this system.
		ExecutorService exec 	= Executors.newFixedThreadPool(processors);		// Set up parallel computing using all cores.

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
		ExecutorService exec2 	= Executors.newFixedThreadPool(processors);		// Set up parallel computing using all cores.

		List<Callable<double[]>> tasksSmaller = new ArrayList<Callable<double[]>>();	// Preallocate.
		for(int shiftX = shift[0]-CoarseStep-1; shiftX < shift[0]+CoarseStep+1; shiftX ++){
			for(int shiftY = shift[1]-CoarseStep-1; shiftY < shift[1]+CoarseStep+1; shiftY ++){
				final int[] shiftEval = {shiftX,shiftY};				
				Callable<double[]> c = new Callable<double[]>() {					// Computation to be done.
					@Override
					public double[] call() throws Exception {						
						double[] Correlation = {
								Correlation(shiftEval),
								shiftEval[0],
								shiftEval[1]};						
						return Correlation;						// Actual call for each parallel process.
					}
				};
				tasksSmaller.add(c);														// Que this task.
			} 
		}
		int[] fineShift = shift;
		try {
			List<Future<double[]>> parallelCompute = exec2.invokeAll(tasksSmaller);				// Execute computation.
			double[] Corr;
			double maxCorr = 0;

			for (int i = 1; i < parallelCompute.size(); i++){							// Loop over and transfer results.
				try {
					Corr = parallelCompute.get(i).get();			
					if (Corr[0] > maxCorr){
						maxCorr = Corr[0];
						fineShift[0] = (int) Corr[1];
						fineShift[1] = (int) Corr[2];

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
			exec2.shutdown();
		}
		return fineShift;
	}

	public static void main(String[] args){
		ShortProcessor IP = new ShortProcessor(2560,2560);		
		IP.set(0);
		IP.set(500, 500, 10000);
		IP.set(250, 250, 10000);
		IP.set(750, 750, 10000);
		IP.set(350, 350, 10000);
		IP.blurGaussian(3); // 3 pixel search radius.
		int[][] A = IP.getIntArray();
		ShortProcessor IP2 = new ShortProcessor(2560,2560);	
		IP2.set(0);
		IP2.set(540, 482, 10000);
		IP2.set(290, 232, 10000);
		IP2.set(790, 732, 10000);
		IP2.set(390, 332, 10000);
		IP2.blurGaussian(3); // 3 pixel search radius.
		int[][] B = IP2.getIntArray();

		autoCorr2D CorrCalc = new autoCorr2D(A, B, 2560,2560); // Setup calculations.

		int[] maxshift = {50,50};
		long start = System.nanoTime();										// Timer.
		int[] ShiftP = CorrCalc.optimizeParallel(maxshift);				// optimize.
		long elapsed = System.nanoTime() - start;
		long startnorm = System.nanoTime();
		int[] Shift = CorrCalc.optimize(maxshift);
		long stopnorm = System.nanoTime();
		int sum = (int) ((stopnorm-startnorm)/1000000);
		elapsed /= 1000000;
		System.out.println(String.format("Elapsed time: %d ms", elapsed));
		System.out.println(String.format("... but compute tasks waited for total of %d ms; speed-up of %.2fx", sum, sum / (elapsed * 1d)));		
	}


}