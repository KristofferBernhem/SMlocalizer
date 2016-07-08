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
 * 2D elliptical gaussian function, based on code by Yoshiyuki Arai.
 * Uses Apache Commons Math3 for optimization through Levenberg-Marquardt.
 * Derivation of jacobian and function taken from PALMsiever.
 * 
 * v1.0.0 Kristoffer Bernhem
 */

import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

public class Gauss2D {

	/* 
	 * Input:
	 * V[0] = Amplitude
	 * V[1] = x0;
	 * V[2] = y0;
	 * V[3] = sigma x;
	 * V[4] = sigma y;
	 * V[5] = offset
	 * V[6] = theta.
	 * 
	 * Function
	 * xprime = (x - x0)*cos(theta) - (y - y0)*sin(theta)
	 * yprime = (x - x0)*sin(theta) + (y - y0)*cos(theta)
	 * f(x,y) = Amp*exp(-[xrpime^2/2*sigma_x^2 + yprime^2/2*sigma_x^2]) + Background
	 * Jacobian
	 * df/dV(0) = exp(-[xrpime^2/2*sigma_x^2 + yprime^2/2*sigma_x^2])
	 * df/dV(1) = f(x,y)*[(xprime*cos(theta)/sigma_x^2)*(yprime*sin(theta)/sigma_y^2)]
	 * df/dV(2) = f(x,y)*[(-xprime*sin(theta)/sigma_x^2)*(yprime*cos(theta)/sigma_y^2)]
	 * df/dV(3) = f(x,y)*xprime^2/sigma_x^3
	 * df/dV(4) = f(x,y)*yprime^2/sigma_y^3
	 * df/dV(5) = 1;
	 * df/dV(6) = -f(x,y)*[(-xprime/sigma_x^2 *( (x-x0)*sin(theta) + (y-y0)*cos(theta) ) + 
	 * 				yprime/sigma_y^2 * ( (x-x0)*cos(theta) - (y-y0)*sin(theta) ) ] 
	 * 

	 */

	int width; // Width of input data.
	int size; // size of input data.

	public Gauss2D (int width, int size){
		this.width = width;
		this.size = size;			
	}

	/*
	 * Return function evaluation.
	 */
	public MultivariateVectorFunction retMVF (){
		return new MultivariateVectorFunction() {	
			@Override
			public double[] value(double[] V)
					throws IllegalArgumentException {
				double[] eval = new double[size];
				double SigmaX2 = 2*V[3]*V[3];
				double SigmaY2 = 2*V[4]*V[4];
				
//				double x0 = V[1]*Math.cos(V[5]) - V[2]*Math.sin(V[5]);
//				double y0 = V[1]*Math.sin(V[5]) - V[2]*Math.cos(V[5]);
				for (int i = 0; i < eval.length; i++){
					int xi = i % width;
					int yi = i / width;				
					double xprime = (xi - V[1])*Math.cos(V[6]) - (yi - V[2])*Math.sin(V[6]);
					double yprime = (xi - V[1])*Math.sin(V[6]) + (yi - V[2])*Math.cos(V[6]);
					eval[i] = V[0]*Math.exp(-(xprime*xprime/SigmaX2 + yprime*yprime/SigmaY2))+ V[5];
		        	


				}			
				return eval;

			}
		};
	}

	public MultivariateMatrixFunction retMMF() {
		return new MultivariateMatrixFunction() {

			//	@Override
			@Override
			public double[][] value(double[] point)
					throws IllegalArgumentException {
				return jacobian(point);
			}

			private double[][] jacobian(double[] V) {
				double[][] jacobian = new double[size][7];				
				double SigmaX2 = V[3]*V[3]; 	// sigma_x^2
				double SigmaY2 = V[4]*V[4]; 	// sigma_y^2
				double SigmaX3 = SigmaX2*V[3]; 	// sigma_x^3
				double SigmaY3 = SigmaY2*V[4]; 	// sigma_y^3
			//	double x0 = V[1]*Math.cos(V[6]) - V[2]*Math.sin(V[6]);
//				double y0 = V[1]*Math.sin(V[6]) - V[2]*Math.cos(V[6]);
				for (int i = 0; i < jacobian.length; i++){
					int xi = i % width;
					int yi = i / width;
					double xprime = (xi - V[1])*Math.cos(V[6]) - (yi - V[2])*Math.sin(V[6]);
					double yprime = (xi - V[1])*Math.sin(V[6]) + (yi - V[2])*Math.cos(V[6]);
					jacobian[i][0] = Math.exp(-(xprime*xprime/SigmaX2 + yprime*yprime/SigmaY2));
					jacobian[i][1] = V[0]*jacobian[i][0]*((xprime*Math.cos(V[6]/SigmaX2)) + (yprime*Math.sin(V[6])/SigmaY2));
					jacobian[i][2] = V[0]*jacobian[i][0]*((yprime*Math.cos(V[6]/SigmaY2)) - (xprime*Math.sin(V[6])/SigmaX2));
					jacobian[i][3] = V[0]*jacobian[i][0]*xprime*xprime/(SigmaX3);
					jacobian[i][4] = V[0]*jacobian[i][0]*yprime*yprime/(SigmaY3);
					jacobian[i][5] = 1;
					jacobian[i][6] = -V[0]*jacobian[i][0]*
							(((-xprime/SigmaX2)*((xi-V[1])*Math.sin(V[6])) + (yi - V[2])*Math.cos(V[6])) + 
									(yprime/SigmaY2)*((xi-V[1])*Math.cos(V[6])) - (yi - V[2])*Math.sin(V[6]));
		      
				}
				return jacobian;
			}
		};
	}

}
