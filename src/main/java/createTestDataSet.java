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

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;

public class createTestDataSet {
	public static void stack(int width, int height, int frames, int particleCount, int particlePerFrame){
		ImageStack Stack = new ImageStack(width, height);
		ArrayList<Particle> ParticleList = new ArrayList<Particle>();
		double[] xCoord = correctDrift.interp(500, (double) (width-5)*100, (width-10)*100);
		double[] yCoord = correctDrift.interp(500, (double) (height-5)*100, (height-10)*100);

		Random r = new Random();
		for (int i = 0; i < particleCount; i++){  // Create particles.
			Particle nextParticle = new Particle();
			nextParticle.x = xCoord[(int) Math.round(r.nextDouble()*xCoord.length)];
			nextParticle.y = yCoord[(int) Math.round(r.nextDouble()*yCoord.length)];
			nextParticle.sigma_x = 100 + r.nextDouble()*150;
			nextParticle.sigma_y = 100 + r.nextDouble()*150;
			ParticleList.add(nextParticle);
		}
		for (int Frame = 1; Frame <= frames; Frame++){
			ByteProcessor IP  = new ByteProcessor(width,height);
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
				double[] P = new double[7];				
				P[0] = 100 + r.nextDouble()*100;
				P[1] = 3; // add drift.
				P[2] = 3; // add drift.
				P[3] = currParticle.sigma_x/100;
				P[4] = currParticle.sigma_y/100;
				P[5] = Math.PI*r.nextDouble()/2;
				P[6] =0;
				int[] data = GaussBlur(P,7,49);
				for (int j = 0; j < data.length; j++){
					int x = j % 7;
					int y = j / 7;
					x += (int) currParticle.x/100 - 3;
					y += (int) currParticle.y/100 - 3;
					IP.set(x,y, (IP.get(x,y) + data[j]));
				}

			}
			Stack.addSlice(IP);

		}
		System.out.println(Stack.getSize());
		ImagePlus image = new ImagePlus("",Stack);
		
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
