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

import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;


/*
 * Shift-Accumulation 2D autocorrelation.
 */
/* TODO
 * Test effects of varied gaussian blur on results.
 * Create 3D blur. 
 * To slow, not functional in current state.
 */
public class autoCorr2D {
	int[][][] imageAlpha, imageBeta;
	int width, height, depth;

	autoCorr2D(int[][][] imageAlpha, int[][][] imageBeta, int width, int height, int depth){
		this.imageAlpha = imageAlpha;
		this.imageBeta = imageBeta;
		this.width = width;
		this.height = height;
		this.depth = depth;
	}

	autoCorr2D(ArrayList<Particle> imageAlpha, ArrayList<Particle> imageBeta, int width, int height, int depth, int[] scale){
		int[][][] ImAlpha = new int[width][height][depth];
		int[][][] ImBeta = new int[width][height][depth];
		ImageStack Stack_Alpha = new ImageStack(width,height,depth);
		for (int Frame = 1; Frame <= depth; Frame ++){ // Initialize.
			ImageProcessor IP = Stack_Alpha.getProcessor(Frame);
			IP.set(0); // Set to 0.
		}


		for (int i = 0; i< imageAlpha.size(); i++){
			ImageProcessor IP = Stack_Alpha.getProcessor((int)Math.round(imageAlpha.get(i).z/scale[2])+1); // Set correct slize.
			IP.set(
					(int)Math.round(imageAlpha.get(i).x/scale[0]), 
					(int)Math.round(imageAlpha.get(i).y/scale[1]), 
					10000);
		}
		ImageStack Stack_Beta = new ImageStack(width,height,depth);
		for (int Frame = 1; Frame <= depth+1; Frame ++){ // Initialize.
			ImageProcessor IP = Stack_Beta.getProcessor(Frame);
			IP.set(0); // Set to 0.
		}


		for (int i = 0; i< imageBeta.size(); i++){
			ImageProcessor IP = Stack_Beta.getProcessor((int)Math.round(imageBeta.get(i).z/scale[2])+1); // Set correct slize.
			IP.set(
					(int)Math.round(imageBeta.get(i).x/scale[0]), 
					(int)Math.round(imageBeta.get(i).y/scale[1]), 
					10000);
		}
		for (int Frame = 1; Frame <= depth; Frame ++){ // Initialize.
			ImageProcessor IP = Stack_Alpha.getProcessor(Frame);
			IP.blurGaussian(3); 	// 3 pixel search radius.
			int[][] temp = IP.getIntArray();
			for (int i = 0; i < width; i++){
				for (int j = 0; j < height; j++){
					ImAlpha[i][j][Frame-1] = temp[i][j];
				}
			}

			ImageProcessor IP2 = Stack_Beta.getProcessor(Frame);
			IP2.blurGaussian(3); 	// 3 pixel search radius.
			int[][] temp2 = IP2.getIntArray();
			for (int i = 0; i < width; i++){
				for (int j = 0; j < height; j++){
					ImBeta[i][j][Frame-1] = temp2[i][j];
				}
			}
		}
		this.imageAlpha = ImAlpha;
		this.imageBeta =  ImBeta;
		this.width = width;
		this.height = height;
		this.depth = depth;
	}

	public double Correlation(int[] shift){	
		/*
		 * Limit loop as 0 padding will result in the same value as exclusion.
		 */
		int minX = 0;
		int maxX = width;
		int minY = 0;
		int maxY = height;
		int minZ = 0;
		int maxZ = depth-1;
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
		if (shift[2] < 0){
			maxZ = depth + shift[2];
		}else if(shift[2] > 0){
			minZ = shift[2];
		}
		double Corr = 0;

		for(int i = minX; i < maxX; i ++){
			for (int j = minY; j < maxY; j++){
				for (int k = minZ; k < maxZ; k++){					
					Corr += imageAlpha[i][j][k]*imageBeta[i-shift[0]][j-shift[1]][k-shift[2]];
				}
			}

		}
		return Corr;
	}

	public int[] optimize(int[] maxShift){
		int[] CoarseStep = {5,5,1};
		int[] shift = new int[3];
		double[][] Correlation = new double[maxShift[0]*2*2*maxShift[1]*2*(1+maxShift[2])][4];
		int count = 0;		
		for(int shiftX = -maxShift[0]; shiftX <= maxShift[0]; shiftX += CoarseStep[0]){
			for(int shiftY = -maxShift[1]; shiftY <= maxShift[1]; shiftY += CoarseStep[1]){
				for(int shiftZ = -maxShift[2]; shiftZ <= maxShift[2]; shiftZ += CoarseStep[2]){
					int[] shiftEval = {shiftX,shiftY,shiftZ};
					Correlation[count][0] = Correlation(shiftEval);
					Correlation[count][1] = shiftX;
					Correlation[count][2] = shiftY;
					Correlation[count][3] = shiftZ;
					count++;
				}	
			}		
		}

		// done in this way for parallelization implementation later on.
		double minCorr = Correlation[0][0];
		shift[0] = (int) Correlation[0][1];
		shift[1] = (int) Correlation[0][2];
		shift[2] = (int) Correlation[0][3];
		for (int index = 1; index < Correlation.length; index++){
			if(Correlation[index][0] > minCorr){
				minCorr = Correlation[index][0];
				shift[0] = (int) Correlation[index][1];
				shift[1] = (int) Correlation[index][2];
				shift[2] = (int) Correlation[index][3];

			}
		}

		double[][] CorrelationSmaller = new double[((CoarseStep[0]+1)*2+1)*((CoarseStep[1]+1)*2+1)*((CoarseStep[2]+1)*2+1)][4];
		count = 0;
		for(int shiftX = shift[0]-CoarseStep[0]; shiftX <= shift[0]+CoarseStep[0]; shiftX ++){
			for(int shiftY = shift[1]-CoarseStep[1]; shiftY <= shift[1]+CoarseStep[1]; shiftY ++){
				for(int shiftZ = shift[1]-CoarseStep[2]; shiftZ <= shift[1]+CoarseStep[2]; shiftZ ++){
					int[] shiftEval = {shiftX,shiftY,shiftZ};
					CorrelationSmaller[count][0] = Correlation(shiftEval);
					CorrelationSmaller[count][1] = shiftX;
					CorrelationSmaller[count][2] = shiftY;
					CorrelationSmaller[count][3] = shiftZ;
					count++;
				}			
			}
		}

		// done in this way for parallelization implementation later on.
		for (int index = 1; index < CorrelationSmaller.length; index++){
			if(CorrelationSmaller[index][0] > minCorr){
				minCorr = CorrelationSmaller[index][0];
				shift[0] = (int) CorrelationSmaller[index][1];
				shift[1] = (int) CorrelationSmaller[index][2];
				shift[2] = (int) CorrelationSmaller[index][3];

			}
		}
		return shift;
	}

	public int[] optimizeParallel(int[] maxShift){
		int[] CoarseStep = {5,5,1};
		int[] shift = new int[3];
		List<Callable<double[]>> tasks = new ArrayList<Callable<double[]>>();	// Preallocate.
		for(int shiftX = -maxShift[0]; shiftX <= maxShift[0]; shiftX += CoarseStep[0]){
			for(int shiftY = -maxShift[1]; shiftY <= maxShift[1]; shiftY += CoarseStep[1]){
				for(int shiftZ = -maxShift[2]; shiftZ <= maxShift[2]; shiftZ += CoarseStep[2]){
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
		ExecutorService exec2 	= Executors.newFixedThreadPool(processors);		// Set up parallel computing using all cores.

		List<Callable<double[]>> tasksSmaller = new ArrayList<Callable<double[]>>();	// Preallocate.
		for(int shiftX = shift[0]-CoarseStep[0]; shiftX <= shift[0]+CoarseStep[0]; shiftX ++){
			for(int shiftY = shift[1]-CoarseStep[1]; shiftY <= shift[1]+CoarseStep[1]; shiftY ++){
				for(int shiftZ = shift[2]-CoarseStep[2]; shiftZ <= shift[2]+CoarseStep[2]; shiftZ ++){
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
					tasksSmaller.add(c);														// Que this task.
				} 
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
						fineShift[2] = (int) Corr[3];

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
		ShortProcessor zeroIP = new ShortProcessor(2560,2560);	
		zeroIP.set(0);
		ShortProcessor IP2 = new ShortProcessor(2560,2560);	
		int[][][] A = new int[2560][2560][50];
		ShortProcessor IP = new ShortProcessor(2560,2560);		
		IP.set(0);
		IP.set(500, 500, 10000);
		IP.set(250, 250, 10000);
		IP.set(750, 750, 10000);
		IP.set(350, 350, 10000);
		IP.blurGaussian(3); // 3 pixel search radius.
		int[][] slice = zeroIP.getIntArray();
		for(int Frame = 0; Frame < 50; Frame ++){
			
			for (int i = 0; i < 2560; i++){
				for (int j = 0; j < 2560; j++){
					A[i][j][Frame] = slice[i][j];
				}
			}
		}
		int[][] Aslice = IP.getIntArray();
		for (int i = 0; i < 2560; i++){
			for (int j = 0; j < 2560; j++){
				A[i][j][30] = Aslice[i][j];
			}
		}



		IP2.set(0);
		IP2.set(540, 482, 10000);
		IP2.set(290, 232, 10000);
		IP2.set(790, 732, 10000);
		IP2.set(390, 332, 10000);
		int[][][] B = new int[2560][2560][50];
		IP2.blurGaussian(3); // 3 pixel search radius.
		for(int Frame = 0; Frame < 50; Frame ++){
			//slice = zeroIP.getIntArray();
			for (int i = 0; i < 2560; i++){
				for (int j = 0; j < 2560; j++){
					B[i][j][Frame] = slice[i][j];
				}
			}
		}
		

		int[][] Bslice = IP2.getIntArray();
		for (int i = 0; i < 2560; i++){
			for (int j = 0; j < 2560; j++){
				B[i][j][33] = Bslice[i][j];
			}
		}


		autoCorr2D CorrCalc = new autoCorr2D(A, B, 2560,2560,50); // Setup calculations.

		int[] maxshift = {50,50,20};
		long start = System.nanoTime();										// Timer.
		int[] ShiftP = CorrCalc.optimizeParallel(maxshift);				// optimize.
		long elapsed = System.nanoTime() - start;
		long startnorm = System.nanoTime();
//		int[] Shift = CorrCalc.optimize(maxshift);
		long stopnorm = System.nanoTime();
		int sum = (int) ((stopnorm-startnorm)/1000000);
		elapsed /= 1000000;
		System.out.println(String.format("Elapsed time: %d ms", elapsed));
		System.out.println(String.format("... but compute tasks waited for total of %d ms; speed-up of %.2fx", sum, sum / (elapsed * 1d)));
		System.out.println(ShiftP[0]+" "+ShiftP[1]+" "+ShiftP[2]);
	}


}