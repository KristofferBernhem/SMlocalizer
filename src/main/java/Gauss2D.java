/*
* 2D elliptical gaussian function, based on code by Yoshiyuki Arai (https://github.com/arayoshipta/TwoDGaussainProblemExample).
* Uses Apache Commons Math3 for optimization through Levenberg-Marquardt.
* 
* v0.1.0 Kristoffer Bernhem
*/
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

public class TwoDRotGaussFcn {
	
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
	 * a = cos^2(theta)/2*sigmaX^2 + sin^2(theta)/2*sigmaY^2
	 * b = -sin(2*theta)/4*sigmaX^2 + sin(2*theta)/4*sigmaY^2
	 * c = sin^2(theta)/2*sigmaX^2 + cos^2(theta)/2*sigmaY^2
	 * f(x,y) = Amp*exp[-{a(x-x0)^2 - 2b(x-x0)(y-y0) + c(y-y0)^2}]
	 * 
	 * Jacobian: 
	 */
	
	int width; // Width of input data.
	int size; // size of input data.
	
	public TwoDRotGaussFcn (int width, int size){
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
				double SigmaX2 = V[3]*V[3];
				double SigmaY2 = V[4]*V[4];
				double a = (Math.cos(V[6])*Math.cos(V[6])/(2*SigmaX2)) + (Math.sin(V[6])*Math.sin(V[6])/(2*SigmaY2));
				double b = (-Math.sin(2*V[6])/(4*SigmaX)) + (Math.sin(2*V[6])/(4*SigmaY));
				double c = (Math.sin(V[6])*Math.cos(V[6])/(2*SigmaX2)) + (Math.cos(V[6])*Math.sin(V[6])/(2*SigmaY2));
				double Amp = V[0]/Math.sqrt(2*Math.PI*SigmaX2*SigmaY2);
				for (int i = 0; i < eval.length; i++){
					int xi = i % width;
					int yi = i / width;
					double A = a*(xi - V[1])*(xi - V[1]);
					double B = 2*b*(xi - V[1])(yi - V[2]);
					double C = c*(yi - V[2])*(yi - V[2]);
					eval[i] = Amp*Math.exp(-(A-B+C))+V[5];
				}
				return eval;
			}
		}
	}
	/*
	 * Return Jacobian of function.
	 */
    public MultivariateMatrixFunction retMMF() {
    	return new MultivariateMatrixFunction() {

			@Override
			public double[][] value(double[] point)
					throws IllegalArgumentException {
				return jacobian(point);
			}
			
			private double[][] jacobian(double[] v) {
				double[][] jacobian = new double[size][7];
				double SigmaX2 = V[3]*V[3];
				double SigmaY2 = V[4]*V[4];
				double SqrtSigma = Math.sqrt(2*Math.PI*SigmaX2*SigmaY2);
				double a = (Math.cos(V[6])*Math.cos(V[6])/(2*SigmaX2)) + (Math.sin(V[6])*Math.sin(V[6])/(2*SigmaY2));
				double b = (-Math.sin(2*V[6])/(4*SigmaX)) + (Math.sin(2*V[6])/(4*SigmaY));
				double c = (Math.sin(V[6])*Math.cos(V[6])/(2*SigmaX2)) + (Math.cos(V[6])*Math.sin(V[6])/(2*SigmaY2));
				for (int i = 0; i < jacobian.length; i ++){
					double A = a*(xi - V[1])*(xi - V[1]);
					double B = 2*b*(xi - V[1])(yi - V[2]);
					double C = c*(yi - V[2])*(yi - V[2]);
					int xi = i % width;
		        	int yi = i / width;
					jacobian[i][0] = -(A-B+C)/SqrtSigma;
					// The onees below here needs to be edited and checked for elliptical 2d gausses.
		            jacobian[i][1] = v[0]*(xi-v[1])/v3v3*jacobian[i][0]; 						//df(x,y)/dv1
		            jacobian[i][2] = v[0]*(yi-v[2])/v4v4*jacobian[i][0]; 						//df(x,y)/dv2
		            jacobian[i][3] = jacobian[i][1]*(xi-v[1])/v[3]-v[0]*jacobian[i][0]/v[3]; 	//df(x,y)/dv3
		            jacobian[i][4] = jacobian[i][2]*(yi-v[2])/v[4]-v[0]*jacobian[i][0]/v[4];	//df(x,y)/dv4
		            jacobian[i][5] = 1;															//df(x,y)/dv5
		            jacobian[i][6] = 0; 														//df(x,y)/dv6
				}
				
				return jacobian
			}
			
			
		}
   	}
}