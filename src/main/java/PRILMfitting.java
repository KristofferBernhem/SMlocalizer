import java.util.ArrayList;

import ij.ImagePlus;
import ij.WindowManager;
import ij.process.ImageStatistics;

public class PRILMfitting {
	public static ArrayList<Particle> fit(ArrayList<Particle> inputResults)
	{
		ArrayList<Particle> results = new ArrayList<Particle>();
		double[][] calibration = getCalibration();
		int distance = 	(int)ij.Prefs.get("SMLocalizer.calibration.PRILM.distance",0);
		distance *= distance;
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
		int nChannels = image.getNChannels();
		int[] totalGain 		= {100};
		int selectedModel 		= 0; // CPU
		double meanRsquare 		= 0;
		int calibrationLength 	= 0;
		boolean[][] include = new boolean[7][10];
		double[][] lb 		= new double[7][10];
		double[][] ub 		= new double[7][10];
		for (int i = 0; i < 10; i++)
		{
			include[0][i] 		= false;
			include[1][i] 		= false;
			include[2][i] 		= false;
			include[3][i] 		= true;
			include[4][i] 		= false;
			include[5][i] 		= false;
			include[6][i] 		= false;

			lb[0][i]			= 0;
			lb[1][i]			= 0;
			lb[2][i]			= 0;
			lb[3][i]			= 0.8;
			lb[4][i]			= 0;
			lb[5][i]			= 0;
			lb[6][i]			= 0;

			ub[0][i]			= 0;
			ub[1][i]			= 0;
			ub[2][i]			= 0;
			ub[3][i]			= 1.0;
			ub[4][i]			= 0;
			ub[5][i]			= 0;
			ub[6][i]			= 0;
		}
		int finalWindow = 0;
		double finalLevel = 0;
		double finalSigma = 0;
		int finalDist = 0;
		for (int window = 5; window <= 7; window +=2)
		{
			for (double level = 0.7; level > 0.4; level -= 0.1)
			{
				for (double maxSigma = 2.5; maxSigma < 4; maxSigma += 0.5)
				{
					for (int maxDist = 650; maxDist < 1000; maxDist += 200)
					{
						int maxSqdist = maxDist * maxDist;
						ImageStatistics IMstat 	= image.getStatistics(); 
						int gWindow 			= window;
						int[] MinLevel 			= {(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level),(int) (IMstat.max*level)};
						int minPosPixels 		= window*window-5;

						localizeAndFit.run(MinLevel, inputPixelSize, totalGain, selectedModel, maxSigma,"PRILM");
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
								finalWindow = window;
								finalLevel = level;
								finalSigma = maxSigma;
								finalDist = maxDist;
							}

						}
					} // iterate over maxDistance
				} // iterate over maxSigma
			} // iterate over level.
		} // iterate window size.
		int maxSqdist = finalDist * finalDist;
		ImageStatistics IMstat 	= image.getStatistics(); 
		int[] gWindow 			= {finalWindow,finalWindow,finalWindow,finalWindow,finalWindow,finalWindow,finalWindow,finalWindow,finalWindow,finalWindow}; // iterate.
		int[] MinLevel 			= {(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel),(int) (IMstat.max*finalLevel)};
		int[] minPosPixels 		= {finalWindow*finalWindow-5,finalWindow*finalWindow-5,finalWindow*finalWindow-5,finalWindow*finalWindow-5,finalWindow*finalWindow-5,finalWindow*finalWindow-5,finalWindow*finalWindow-5,finalWindow*finalWindow-5,finalWindow*finalWindow-5,finalWindow*finalWindow-5};
		
		localizeAndFit.run(MinLevel, inputPixelSize, totalGain, selectedModel, finalSigma,"PRILM");
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


		/*
		 * STORE calibration file:
		 */
		ij.Prefs.set("SMLocalizer.calibration.PRILM.window",finalWindow);
		ij.Prefs.set("SMLocalizer.calibration.PRILM.sigma",finalSigma);
		ij.Prefs.set("SMLocalizer.calibration.PRILM.distance",finalDist);

		ij.Prefs.set("SMLocalizer.calibration.PRILM.height",calibration.length);
		ij.Prefs.set("SMLocalizer.calibration.PRILM.channels",calibration[0].length);
		ij.Prefs.set("SMLocalizer.calibration.PRILM.step",zStep);
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.PRILM.Ch"+Ch+"."+i,calibration[i][Ch-1]);
			}
		} 
		ij.Prefs.savePreferences(); // store settings. 
/*		double[] printout = new double[calibration.length];
		for (int i = 0; i < printout.length; i++)
			printout[i] = calibration[i][0];
		correctDrift.plot(printout);
*/

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
		
		z *= ij.Prefs.get("SMLocalizer.calibration.PRILM.step",0); // scale.
		return z;
	}
	public static double[][] getCalibration()
	{
		int nChannels = (int)ij.Prefs.get("SMLocalizer.calibration.PRILM.channels",0);
		double[][] calibration = new double[(int)ij.Prefs.get("SMLocalizer.calibration.PRILM.height",0)][nChannels];
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				calibration[i][Ch-1] = ij.Prefs.get("SMLocalizer.calibration.PRILM.Ch"+Ch+"."+i,0);
			}
		} 

		return calibration;
	}
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
