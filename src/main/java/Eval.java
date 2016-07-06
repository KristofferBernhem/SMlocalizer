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
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;


/**
 * This is the test program for two dimensional Gaussian function fitting.
 * @author araiyoshiyuki
 * @date 08/01/2015
 */
public class Eval {
	
	private double[] data;
	private double[] newStart;
	private int data_width;
	private int[] optim_param;
	private LeastSquaresProblem lsp;

	/**
	 * @param data	input data
	 * @param newStart	initial values
	 * @param data_width	data width
	 * @param optim_param	[0]:maxEvaluation, [1]:maxIteration
	 */
	public Eval(double[] data, double[] newStart, int data_width, int[] optim_param) {
		this.data = data;
		this.newStart = newStart;
		this.data_width = data_width;
		this.optim_param = optim_param;
		buildlsb();
	}
	
	/**
	 * build LeastSquareProblem by using constructer data
	 */
	private void buildlsb() {
		//construct two-dimensional Gaussian function
		Gauss2D tdgf = new Gauss2D(this.data_width,this.data.length);
		
		//prepare construction of LeastSquresProblem by builder
		LeastSquaresBuilder lsb = new LeastSquaresBuilder();

		//set model function and its jacobian
		lsb.model(tdgf.retMVF(), tdgf.retMMF());
		//set target data
		lsb.target(this.data);
		//set initial parameters
		lsb.start(this.newStart);
		//set upper limit of evaluation time
		lsb.maxEvaluations(this.optim_param[0]);
		//set upper limit of iteration time
		lsb.maxIterations(this.optim_param[1]);
		
		lsp = lsb.build();
	}
	
	/**
	 * Do two dimensional Gaussian fit
	 * @return return the fitted data as Optimum
	 */
	public Optimum fit2dGauss() {
		LevenbergMarquardtOptimizer lmo = new LevenbergMarquardtOptimizer();
		LeastSquaresOptimizer.Optimum lsoo = lmo.optimize(lsp);

		return lsoo;	
	}

	public static void main(String[] args) {						
        //entry the data (5x5)
		double[] inputdata = {
				0  ,12 ,25 ,12 ,0  ,
				12 ,89 ,153,89 ,12 ,
				25 ,153,255,153,25 ,
				12 ,89 ,153,89 ,12 ,
				0  ,12 ,25 ,12 ,0  ,
		};		
		
		//set initial parameters
		double[] newStart = {
				255,
				1,
				1,
				1,
				1,
				1,
				0
		};
		
		Eval tdgp = new Eval(inputdata, newStart, 6, new int[] {1000,100});
		
		try{
			//do LevenbergMarquardt optimization and get optimized parameters
			Optimum opt = tdgp.fit2dGauss();
			final double[] optimalValues = opt.getPoint().toArray();
			
			//output data
			System.out.println("v0: " + optimalValues[0]);
			System.out.println("v1: " + optimalValues[1]);
			System.out.println("v2: " + optimalValues[2]);
			System.out.println("v3: " + optimalValues[3]);
			System.out.println("v4: " + optimalValues[4]);
			System.out.println("v5: " + optimalValues[5]);
			System.out.println("Iteration number: "+opt.getIterations());
			System.out.println("Evaluation number: "+opt.getEvaluations());
		} catch (Exception e) {
			System.out.println(e.toString());
		}
	}

}