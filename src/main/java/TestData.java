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

public class TestData {
	public static ArrayList<Particle> generate(int nFrames, int width, int height, int perFrame, int total){
		ArrayList<Particle> newDataset = new ArrayList<Particle>();
		Random r = new Random();
		double[] X = new double[total];
		double[] Y = new double[total];
		
		for (int i = 0; i < total; i++){
			X[i] =  0.8*width*r.nextDouble();
			Y[i] =  0.8*height*r.nextDouble();
		}
		int count = 0;
		for (int frame = 1; frame <= nFrames; frame++){			
			for (int j = 0; j < perFrame; j++){
				newDataset.add( new Particle(
						X[count]  + 50* r.nextDouble() + 0.1*width, // x.
						Y[count]  + 50* r.nextDouble() + 0.1*height, // y.
						0,
						frame,						// frame.
						1,
						100*(1 + r.nextDouble()),			// sigma x.
						100*(1 + r.nextDouble()),     	// sigma y.
						0,
						1 + 5*(10*r.nextDouble()), 	// precision x.
						1 + 5*(10*r.nextDouble()), 	// precision y
						0,
						0.1 + 0.1*r.nextDouble(),  	// chi square.
						100 + 500*r.nextInt(),	// photons.			 						
						1));
				count++;
				if (count == total){
					count = 0; //Reset.
				}
					
			}
		}
		return newDataset;
	}
}
