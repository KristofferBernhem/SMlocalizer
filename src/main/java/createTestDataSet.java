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
import java.util.Random;

//import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;
import ij.process.ShortProcessor;

public class createTestDataSet {
	public static void stack(int width, int height, int frames, int particleCount, int particlePerFrame){
		ImageStack Stack = new ImageStack(width, height);
		ArrayList<Particle> ParticleList = new ArrayList<Particle>();
	//	double[] xCoord = correctDrift.interp(500, (double) (width-5)*100, (width-10)*100);
//		double[] yCoord = correctDrift.interp(500, (double) (height-5)*100, (height-10)*100);
//		double[] xCoord = correctDrift.interp(200, (double) (width-2)*100, (width - 4)*100);		
	//	double[] xCoord = correctDrift.interp(Math.round(width/2)*100-100, Math.round(width/2)*100+100, (width - 4)*100);
	//	double[] yCoord = correctDrift.interp(Math.round(height/2)*100, Math.round(height/2)*100, xCoord.length);
		double[] xCoord = {3125,3125,3125,3125};
		double[] yCoord = xCoord;
		//System.out.println("y: " + yCoord[0] + " to " + yCoord[yCoord.length-1]);
		Random r = new Random();
		for (int i = 0; i < particleCount; i++){  // Create particles.
			Particle nextParticle = new Particle();
			nextParticle.x = xCoord[(int) Math.round(r.nextDouble()*xCoord.length)];
			nextParticle.y = yCoord[(int) Math.round(r.nextDouble()*yCoord.length)];
			nextParticle.sigma_x = 150 + r.nextDouble()*100;
			nextParticle.sigma_y = 150 + r.nextDouble()*100;
			ParticleList.add(nextParticle);			
		//	System.out.println("x: " + nextParticle.x + " y: " + nextParticle.y + " sigma: " + nextParticle.sigma_x + " sigma: " + nextParticle.sigma_y);
		}
		int Channels = 2;
		int Slizes = 1;
		int nFrames = frames/Channels;
		int bitDepth = 16;
		ImagePlus imStack = ij.IJ.createHyperStack("Teststack",
				width,
				height,
				Channels,
				Slizes,
				nFrames,
				bitDepth);
		for (int Frame = 1; Frame <= frames; Frame++){
			ShortProcessor IP  = new ShortProcessor(width,height);					
			for (int x = 0; x < width; x++){
				for (int y = 0; y < height; y++){
					IP.set(x, y, 0);
				}
			}
			for (int i = 0; i < particlePerFrame; i++){
				int index = (int)Math.round(particleCount*r.nextDouble());
				if (index == particleCount){
					index--;
				}
				Particle currParticle = ParticleList.get(index);
				/*
				 * Alternate, generate detailed image, downsample afterwards.
				 */
				int winWidth = 700;
				double[] P = new double[7];				
				P[0] = r.nextDouble()*5;
				P[1] = (winWidth-1)/2;// add drift.
				P[2] = (winWidth-1)/2;// add drift.
				P[3] = currParticle.sigma_x;
				P[4] = currParticle.sigma_y;
				P[5] = Math.PI*r.nextDouble()/10;
				P[6] = r.nextDouble();
				int[] data = GaussBlur(P,winWidth,winWidth*winWidth);
				
				for (int j = 0; j < data.length; j++) {
					int x = j % winWidth;
					int y = j / winWidth;			
					x += currParticle.x - P[1];
					y += currParticle.y - P[2];
		//			x = (int) Math.round(x/100) + 2*((Frame-1) % 2);
		//			y = (int) Math.round(y/100) + 2*((Frame-1) % 2);
					x = (int) Math.round(x/100);
					y = (int) Math.round(y/100);
					IP.set(x,y,(IP.get(x,y) + data[j]));					
				}
				
				/*
				 * works, yields symetry.
				 */
				
	/*			double[] P = new double[7];				
				P[0] = 1000 + r.nextDouble()*60000;
				P[1] = 3; // add drift.
				P[2] = 3; // add drift.
				P[3] = currParticle.sigma_x/100;
				P[4] = currParticle.sigma_y/100;
				P[5] = Math.PI*r.nextDouble()/2;
				P[6] = 0;
				int[] data = GaussBlur(P,7,49);
				for (int j = 0; j < data.length; j++){
					int x = j % 7;
					int y = j / 7;
					x += (int) currParticle.x/100 - 3;
					y += (int) currParticle.y/100 - 3;
					IP.set(x,y, (IP.get(x,y) + data[j]));
				}*/

			}
			Stack.addSlice(IP);			
			imStack.setPosition(1+(Frame-1) % 2, 1, (1+(Frame-1)/2));		
			imStack.setProcessor(IP);
			//System.out.println("channel:_" + (1+(Frame-1) % 2) + " frame; " + (1+(Frame-1)/2));
		}

		Calibration cal = new Calibration(imStack);
		cal.pixelHeight = 100;
		cal.pixelWidth 	= 100;
		cal.setXUnit("nm");
		cal.setYUnit("nm");
		imStack.setCalibration(cal);
		imStack.show();

		ImagePlus image = new ImagePlus("",Stack);	
		cal = new Calibration(image);
		cal.pixelHeight = 100;
		cal.pixelWidth 	= 100;
		cal.setXUnit("nm");
		cal.setYUnit("nm");
		image.setCalibration(cal);
		image.show();
	}
	public static int[] GaussBlur(double[] P, int width, int size){
		int[] eval = new int[size];
		double ThetaA = Math.cos(P[5])*Math.cos(P[5])/(2*P[3]*P[3]) + Math.sin(P[5])*Math.sin(P[5])/(2*P[4]*P[4]); 
		double ThetaB = -Math.sin(2*P[5])/(4*P[3]*P[3]) + Math.sin(2*P[5])/(4*P[4]*P[4]); 
		double ThetaC = Math.sin(P[5])*Math.sin(P[5])/(2*P[3]*P[3]) + Math.cos(P[5])*Math.cos(P[5])/(2*P[4]*P[4]);

		//		double SigmaX2 = 2*P[3]*P[3];
		//		double SigmaY2 = 2*P[4]*P[4];		
		for (int i = 0; i < size; i++){
			int xi = i % width;
			int yi = i / width;	
			eval[i] += P[0]*Math.exp(-(ThetaA*(xi - P[1])*(xi - P[1]) - 
					2*ThetaB*(xi - P[1])*(yi - P[2]) +
					ThetaC*(yi - P[2])*(yi - P[2])
					)) + P[6];		
		}
		return eval;
	}
}
