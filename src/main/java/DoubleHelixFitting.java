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
/**
 *
 * @author kristoffer.bernhem@gmail.com
 */


/*
 * TODO: Error handling if calibration file contains fewer channels than experiment.
 */


// TODO: angle calculated correctly but somehow not passed. Check interpolate function for error.
import java.util.ArrayList;

import ij.ImagePlus;
import ij.WindowManager;
import ij.process.ImageStatistics;

public class DoubleHelixFitting {
	public static ArrayList<Particle> fit(ArrayList<Particle> inputResults)
	{
		ArrayList<Particle> results = new ArrayList<Particle>();
		double[][] calibration 		= getCalibration();
		int distance 				= (int)ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.distance",0);
		distance *= distance; // square max distance between centers.
		for (int i = 0; i < results.size(); i++)
		{
			int j = i+1;
			while (j < inputResults.size() && inputResults.get(i).frame == inputResults.get(j).frame
					&& inputResults.get(i).channel == inputResults.get(j).channel)
			{
				if (((inputResults.get(i).x - inputResults.get(j).x)*(inputResults.get(i).x - inputResults.get(j).x) + 
						(inputResults.get(i).x - inputResults.get(j).x)*(inputResults.get(i).x - inputResults.get(j).x)) < distance)
				{
					short dx 		= (short)(inputResults.get(i).x - inputResults.get(j).x); // diff in x dimension.
					short dy 		= (short)(inputResults.get(i).y - inputResults.get(j).y); // diff in y dimension.
					double angle 	= (Math.atan2(dy, dx));
					// translate angle to z:
					Particle temp 	= new Particle();
					temp.channel 	= inputResults.get(i).channel;
					temp.z 		 	= getZ(calibration, temp.channel, angle);
					temp.frame 	 	= inputResults.get(i).frame;
					temp.photons 	= inputResults.get(i).photons + inputResults.get(j).photons;
					temp.x			= (inputResults.get(i).x + inputResults.get(j).x) / 2;
					temp.y			= (inputResults.get(i).y + inputResults.get(j).y) / 2;
					temp.sigma_x 	= (inputResults.get(i).sigma_x + inputResults.get(j).sigma_x) / 2; 		// fitted sigma in x direction.
					temp.sigma_y 	= (inputResults.get(i).sigma_y + inputResults.get(j).sigma_y) / 2; 		// fitted sigma in x direction.
					temp.sigma_z 	= Math.sqrt(dx*dx + dy*dy); 			// fitted sigma in z direction.
					temp.precision_x= Math.min(inputResults.get(i).precision_x,inputResults.get(j).precision_x); 	// precision of fit for x coordinate.
					temp.precision_y= Math.min(inputResults.get(i).precision_y,inputResults.get(j).precision_y); 	// precision of fit for y coordinate.
					temp.precision_z= temp.sigma_z / Math.sqrt(temp.photons); 			// precision of fit for z coordinate.
					temp.r_square 	= Math.min(inputResults.get(i).r_square,inputResults.get(j).r_square);; 	// Goodness of fit.
					temp.include	= 1; 		// If this particle should be included in analysis and plotted.
					if (temp.z != -1)
						results.add(temp);
				}	
				j++;
			}

		}

		return results;
	}

	public static void calibrate(int inputPixelSize,int zStep)
	{
		ImagePlus image 					= WindowManager.getCurrentImage();
		int nFrames 						= image.getNFrames();
		if (nFrames == 1)
			nFrames 						= image.getNSlices(); 
		if (image.getNFrames() == 1)
		{
			image.setPosition(							
					1,			// channel.
					(int) nFrames/2,			// slice.
					1);		// frame.
		}
		else
		{														
			image.setPosition(
					1,			// channel.
					1,			// slice.
					(int) nFrames/2);		// frame.
		}
		int nChannels 			= image.getNChannels();
		int[] totalGain 		= {100};
		int selectedModel 		= 0; // CPU
		double meanRsquare 		= 0;
		int calibrationLength 	= 0;
		boolean[][] include = new boolean[6][10];
		double[][] lb 		= new double[6][10];
		double[][] ub 		= new double[6][10];
		for (int i = 0; i < 10; i++)
		{
			include[0][i] 		= false;
			include[1][i] 		= false;
			include[2][i] 		= true;
			include[3][i] 		= false;
			include[4][i] 		= false;
			include[5][i] 		= false;


			lb[0][i]			= 0;
			lb[1][i]			= 0;
			lb[2][i]			= 0.8;
			lb[3][i]			= 0;
			lb[4][i]			= 0;
			lb[5][i]			= 0;


			ub[0][i]			= 0;
			ub[1][i]			= 0;
			ub[2][i]			= 1.0;
			ub[3][i]			= 0;
			ub[4][i]			= 0;
			ub[5][i]			= 0;

		}

		double finalLevel = 0;
		double finalSigma = 0;
		int finalDist = 0;
		int gWindow = 5;
		if (inputPixelSize < 100)
		{
			gWindow = (int) Math.ceil(500 / inputPixelSize); // 500 nm wide window.

			if (gWindow%2 == 0)
				gWindow++;	
		}
		int finalGWindow = gWindow;
		int loopC = 0;
		while (loopC < 2)
		{
			gWindow = gWindow + loopC*2; // increase window size each loop.
			for (double level = 0.7; level > 0.4; level -= 0.1)
			{
				for (double maxSigma = 2.5; maxSigma < 4; maxSigma += 0.5)
				{
					for (int maxDist = 850; maxDist < 1600; maxDist += 200)
					{
						int maxSqdist = maxDist * maxDist;
						ImageStatistics IMstat 	= image.getStatistics(); 					
						int[] MinLevel 			= {(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level)};					
						//localizeAndFit.run(MinLevel, gWindow, inputPixelSize, totalGain, selectedModel, maxSigma,"PRILM");
						Fit3D.fit(MinLevel,gWindow,inputPixelSize,totalGain,maxSigma);
						/*
						 * clean out fits based on goodness of fit:
						 */

						cleanParticleList.run(lb,ub,include);
						cleanParticleList.delete();
						ArrayList<Particle> result = TableIO.Load();
						for (int i = 0; i < result.size(); i++)
						{
							result.get(i).z = (result.get(i).frame-1)*zStep;
						}
						TableIO.Store(result);
						result 				= TableIO.Load();
						int id 				= 2;		
						double[][] angle	= new double[nFrames][nChannels];
						double[][] distance = new double[nFrames][nChannels];
						int[][] count 	  	= new int[nFrames][nChannels];
						for (int Ch = 1; Ch <= nChannels; Ch++)
						{
							for (int i = 0; i < result.size(); i++)
							{			
								if (result.get(i).include == 1 && result.get(i).channel == Ch) // if the current entry is within ok range and has not yet been assigned.
								{
									int idx = i + 1;
									while (idx < result.size() && result.get(i).channel == result.get(idx).channel)
									{
										if (result.get(i).frame == result.get(idx).frame && result.get(idx).include == 1)
										{
											if (((result.get(i).x - result.get(idx).x)*(result.get(i).x - result.get(idx).x) +
													(result.get(i).y - result.get(idx).y)*(result.get(i).y - result.get(idx).y)) < maxSqdist)
											{
												result.get(idx).include = id;
												result.get(i).include 	= id;								
												short dx = (short)(result.get(i).x - result.get(idx).x); // diff in x dimension.
												short dy = (short)(result.get(i).y - result.get(idx).y); // diff in y dimension.
												angle[(int)(result.get(i).frame-1)][Ch-1] += (Math.atan2(dy, dx)); // angle between points and horizontal axis.
												if (Math.sqrt(dx*dx + dy*dy) > distance[(int)(result.get(i).frame-1)][Ch-1])
													distance[(int)(result.get(i).frame-1)][Ch-1] = Math.sqrt(dx*dx + dy*dy);
												count[(int)(result.get(i).frame-1)][Ch-1]++;												
											}
										}
										idx ++;
									}    				    				
									id++;
								}
							}
						}
						for (int Ch = 1; Ch <= nChannels; Ch++)
						{
							for(int i = 0; i < count.length; i++)
							{
								if (count[i][Ch-1]>0)
								{
									angle[i][Ch-1] 	/= count[i][Ch-1]; // mean angle for this z-depth.			
								}
							}
						}
						int minLength = 40;			
						double[][] calibration = interpolate(angle, minLength, nChannels);
						if (calibrationLength < calibration.length)
						{
							calibrationLength = calibration.length;
							meanRsquare = 0;
						}
						if (calibrationLength == calibration.length)
						{
							double rsquare = 0;
							for (int i = 0; i < result.size(); i++)
							{
								rsquare +=result.get(i).r_square;
							}
							rsquare /= result.size();
							if (rsquare > meanRsquare)
							{								
								meanRsquare = rsquare;							
								finalLevel = level;
								finalSigma = maxSigma;
								finalDist = maxDist;
								finalGWindow = gWindow;
							}

						}
					} // iterate over maxDistance
				} // iterate over maxSigma
			} // iterate over level.
			loopC++;
		}
		System.out.println("final distance " + finalDist);
		finalDist = 1600;
		int maxSqdist 			= finalDist * finalDist;
		ImageStatistics IMstat 	= image.getStatistics(); 
		int[] MinLevel 			= {(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel)};		
		Fit3D.fit(MinLevel,finalGWindow,inputPixelSize,totalGain,finalSigma);
		//localizeAndFit.run(MinLevel, finalGWindow, inputPixelSize, totalGain, selectedModel, finalSigma,"PRILM");
		/*
		 * clean out fits based on goodness of fit:
		 */


		cleanParticleList.run(lb,ub,include);
		cleanParticleList.delete();
		ArrayList<Particle> result = TableIO.Load();
		for (int i = 0; i < result.size(); i++)
		{
			result.get(i).z = (result.get(i).frame-1)*zStep;
		}
		TableIO.Store(result);
		result = TableIO.Load();
		int id = 2;		
		double[][] angle	= new double[nFrames][nChannels];
		double[][] distance = new double[nFrames][nChannels];
		int[][] count 	  	= new int[nFrames][nChannels];
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < result.size(); i++)
			{			
				if (result.get(i).include == 1 && result.get(i).channel == Ch) // if the current entry is within ok range and has not yet been assigned.
				{
					int idx = i + 1;
					while (idx < result.size() && result.get(i).channel == result.get(idx).channel)
					{
						if (result.get(i).frame == result.get(idx).frame && result.get(idx).include == 1)
						{
							if (((result.get(i).x - result.get(idx).x)*(result.get(i).x - result.get(idx).x) +
									(result.get(i).y - result.get(idx).y)*(result.get(i).y - result.get(idx).y)) < maxSqdist)
							{
								//System.out.println(Math.sqrt(((result.get(i).x - result.get(idx).x)*(result.get(i).x - result.get(idx).x) +
								//	(result.get(i).y - result.get(idx).y)*(result.get(i).y - result.get(idx).y))));
								result.get(idx).include = id;
								result.get(i).include 	= id;								
								short dx = (short)(result.get(i).x - result.get(idx).x); // diff in x dimension.
								short dy = (short)(result.get(i).y - result.get(idx).y); // diff in y dimension.
								angle[(int)(result.get(i).frame-1)][Ch-1] += (Math.atan2(dy, dx)); // angle between points and horizontal axis.
								System.out.println("frame: " + result.get(i).frame + " angle: " + (Math.atan2(dy, dx)));
								count[(int)(result.get(i).frame-1)][Ch-1]++;
								if (Math.sqrt(dx*dx + dy*dy) > distance[(int)(result.get(i).frame-1)][Ch-1])
									distance[(int)(result.get(i).frame-1)][Ch-1] = Math.sqrt(dx*dx + dy*dy);
								

							}
						}
						idx ++;
					}    				    				
					id++;
				}
			}
		}
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for(int i = 0; i < count.length; i++)
			{
				if (count[i][Ch-1]>0)
				{
					angle[i][Ch-1] 	/= count[i][Ch-1]; // mean angle for this z-depth.					
				}
			}
		}
		int minLength = 40;			
		double[][] calibration = interpolate(angle, minLength, nChannels);


		/*
		 * STORE calibration file:
		 */
		System.out.print(calibration.length);
		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.window",finalGWindow);
		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.sigma",finalSigma);
		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.distance",finalDist);

		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.height",calibration.length);
		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.channels",calibration[0].length);
		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.step",zStep);
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.Ch"+Ch+"."+i,calibration[i][Ch-1]);
			}
		} 
		ij.Prefs.savePreferences(); // store settings. 
				double[] printout = new double[calibration.length];
		for (int i = 0; i < printout.length; i++)
			printout[i] = calibration[i][0];
		correctDrift.plot(printout);
		 

	} // calibrate.

	public static double getZ (double[][] calibration, int channel, double angle)
	{
		double z = 0;
		int idx = 0;
		while (calibration[idx][channel-1] < angle && idx < calibration.length)
		{
			idx++;
		}
		if (idx == calibration.length -1 && angle > calibration[idx][channel-1])
			z = -1;
		else if (calibration[idx][channel-1] == angle)
			z = idx;
		else if (calibration[0][channel-1] == angle)
			z = 0;
		else // interpolate
		{
			double diff = calibration[idx][channel-1] - calibration[idx - 1][channel-1];
			double fraction = (angle - calibration[idx - 1][channel-1]) / diff;
			z = idx - 1 + fraction;
		} 					

		z *= ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.step",0); // scale.
		return z;
	}

	// returns calibration[zStep][channel]
	public static double[][] getCalibration()
	{
		int nChannels = (int)ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.channels",0);
		double[][] calibration = new double[(int)ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.height",0)][nChannels];
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				calibration[i][Ch-1] = ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.Ch"+Ch+"."+i,0);
			}
		} 

		return calibration;
	}

	/*
	 * Used in generation of calibration file. Smoothes out fitted results, 5 point moving window mean.
	 */
	public static double[][] interpolate(double[][] result, int minLength, int Channels)
	{
		int start 	= 0;
		int end 	= 0;
		int counter = 0;
		int maxCounter = 0;
		for (int Ch = 1; Ch <= Channels; Ch++)
		{
			for (int i = 0; i < result.length; i++) // loop over all and determine start and end
			{
				if (result[i][Ch-1] < 0)
				{
					counter++;
				}
				if (result[i][Ch-1] >= 0)
				{
					if (counter >= minLength)
						end = i-1;
					counter = 0;

				}
				if (counter == minLength)
				{
					start = i-minLength + 1;
				}
			}
			if ((end-start+1) > maxCounter)
				maxCounter = end-start+1;
		}

		double[][] calibration = new double[maxCounter][Channels];		

		start 	= 0;
		end 	= 0;
		counter = 0;
		for (int Ch = 1; Ch <= Channels; Ch++)
		{
			for (int i = 0; i < result.length; i++) // loop over all and determine start and end
			{
				if (result[i][Ch-1] < 0)
				{
					counter++;
				}
				if (result[i][Ch-1] >= 0)
				{
					if (counter >= minLength)
						end = i-1;
					counter = 0;

				}
				if (counter == minLength)
				{
					start = i-minLength + 1;
				}
			}

			for (int i = 0; i < calibration.length; i++)
			{
				if (i == 0) 
					calibration[i][Ch-1] = (result[start][Ch-1] + result[start + 1][Ch-1] + result[start + 2][Ch-1]) / 3;
				else if (i == 1)
					calibration[i][Ch-1] = (result[start][Ch-1] + result[start + 1][Ch-1] + result[start + 2][Ch-1] + result[start + 3][Ch-1]) / 4;
				else if (i == calibration.length-2)
					calibration[i][Ch-1] = (result[start + i + 1][Ch-1] + result[start + i][Ch-1] + result[start + i - 1][Ch-1] + result[start + i - 2][Ch-1])/4;
				else if (i == calibration.length-1)
					calibration[i][Ch-1] = (result[start + i][Ch-1] + result[start + i - 1][Ch-1]+ result[start + i - 2][Ch-1])/3;
				else
					calibration[i][Ch-1] = (result[start + i][Ch-1] + result[start + i - 1][Ch-1] + result[start + i + 1][Ch-1] + result[start + i - 2][Ch-1] + result[start + i + 2][Ch-1])/5;
			}
		}

		return calibration;
	}
}