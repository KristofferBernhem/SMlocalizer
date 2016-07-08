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

	
public class AutoCorr_2 {
	public static double[] getLambda(ArrayList<Particle> First, ArrayList<Particle> Second, int[] stepSize,int[] lb, int[] ub){
		double width = 0;
		double height = 0;
		double depth = 0;
		// Get max x and y of the datasets.
		for (int i = 0; i < First.size(); i++){
			if (First.get(i).x > width){
				width = First.get(i).x; 
			}
			if (First.get(i).y > height){
				height = First.get(i).x; 
			}
			if (First.get(i).z > depth){
				depth = First.get(i).z; 
			}
		}
		for (int i = 0; i < Second.size(); i++){
			if (Second.get(i).x > width){
				width = Second.get(i).x; 
			}
			if (Second.get(i).y > height){
				height = Second.get(i).x; 
			}
			if (Second.get(i).z > depth){
				depth = Second.get(i).z; 
			}
		}
		for (int i = 0; i < 3; i++){
			lb[i] = (int) Math.round(lb[i]/stepSize[i]);
			ub[i] = (int) Math.round(ub[i]/stepSize[i]);
		}
		
		if (depth == 0){ // 2D data.
			double[] correlation = new double[(ub[0]-lb[0])*(ub[1]-lb[1])];
			double[][] DataArrayFirst = new double[(int) Math.round(width/stepSize[0])+1][(int) Math.round(height/stepSize[1])+1];
			double[][] DataArraySecond = new double[(int) Math.round(width/stepSize[0])+1][(int) Math.round(height/stepSize[1])+1];
		
			for(int shiftX = lb[0]; shiftX <= ub[0]; shiftX++){
				for(int shiftY = lb[0]; shiftY <= ub[0]; shiftY++){
					//correlation[count];
				}
				
			}
		}else{ // 3D data.
			double[][][] DataArrayFirst = new double[(int) Math.round(width/stepSize[0])+1][(int) Math.round(height/stepSize[1])+1][(int) Math.round(depth/stepSize[2])+1];
			double[][][] DataArraySecond = new double[(int) Math.round(width/stepSize[0])+1][(int) Math.round(height/stepSize[1])+1][(int) Math.round(depth/stepSize[2])+1];
		}
		
		
		
	return null;	
	}
	
	public double eval(double[][] Array1, double[][] Array2,int stepX, int stepY){ // Evaluate correlation between Array1 and Array2 with shift stepX and stepY for Array2. Both arrays has to have the same size.
		double correlation = 0;
		
		for (int width = stepX-1; width < Array1.length; width++){
			for (int height = stepY-1; height < Array1[0].length; height++){
				correlation += Array1[width][height]*Array2[width-stepX][height-stepY];
			}
		}
				
		return correlation;
	}
}
