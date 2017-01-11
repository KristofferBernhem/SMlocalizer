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



import java.util.ArrayList;

import ij.ImagePlus;
import ij.WindowManager;
import ij.process.ImageStatistics;

public class PRILMfitting {
	public static ArrayList<Particle> fit(ArrayList<Particle> inputResults)
	{
		ArrayList<Particle> results = new ArrayList<Particle>(); 	// output.
		double[][] calibration 		= getCalibration();				// get calibration table.
		int distance 				= (int)ij.Prefs.get("SMLocalizer.calibration.PRILM.distance",0);
		distance *= distance; 										// square max distance between centers.
		double[][] offset 			= getOffset(); // get xyz offset from channel 1.
		for (int i = 0; i < inputResults.size(); i++)
		{
			int j = i+1;
			while (j < inputResults.size() 												// basic bounds check.
					&& inputResults.get(i).frame == inputResults.get(j).frame    		// while looking within the same frame.
					&& inputResults.get(i).channel == inputResults.get(j).channel)		// while we're looking within the same channel.
			{
				if (((inputResults.get(i).x - inputResults.get(j).x)*(inputResults.get(i).x - inputResults.get(j).x) + 				// if x
						(inputResults.get(i).y - inputResults.get(j).y)*(inputResults.get(i).y - inputResults.get(j).y)) < distance) // and y are close enough.
				{
					short dx 		= (short)(inputResults.get(i).x - inputResults.get(j).x); 				// diff in x dimension.
					short dy 		= (short)(inputResults.get(i).y - inputResults.get(j).y); 				// diff in y dimension.
					double angle 	= (Math.atan2(dy, dx));													// angle between the two lobes.
					// translate angle to z:
					Particle temp 	= new Particle();														
					temp.channel 	= inputResults.get(i).channel;
					temp.z 		 	= getZ(calibration, temp.channel, angle);								// get z-position based on angle and calibration file.
					temp.frame 	 	= inputResults.get(i).frame;
					temp.photons 	= inputResults.get(i).photons + inputResults.get(j).photons;
					temp.x			= (inputResults.get(i).x + inputResults.get(j).x) / 2;
					temp.y			= (inputResults.get(i).y + inputResults.get(j).y) / 2;
					temp.sigma_x 	= (inputResults.get(i).sigma_x + inputResults.get(j).sigma_x) / 2; 				// fitted sigma in x direction.
					temp.sigma_y 	= (inputResults.get(i).sigma_y + inputResults.get(j).sigma_y) / 2; 				// fitted sigma in y direction.
					temp.precision_x= Math.max(inputResults.get(i).precision_x,inputResults.get(j).precision_x); 	// precision of fit for x coordinate.
					temp.precision_y= Math.max(inputResults.get(i).precision_y,inputResults.get(j).precision_y); 	// precision of fit for y coordinate.
					temp.precision_z= 600 / Math.sqrt(temp.photons); 												// precision of fit for z coordinate.
					temp.r_square 	= Math.min(inputResults.get(i).r_square,inputResults.get(j).r_square);; 		// Goodness of fit.
					temp.include	= 1; 																			// If this particle should be included in analysis and plotted.
					if(temp.z != -1 && temp.channel>1) // if within ok z range. For all but first channel, move all particles by x,y,z offset for that channel to align all to first channel.
					{
						temp.x -= offset[0][temp.channel-1];
						temp.y -= offset[1][temp.channel-1];
						temp.z -= offset[2][temp.channel-1];
						results.add(temp);
					}
					
					if (temp.z != -1 && temp.channel==1) // if within ok z range. Don't shift first channel.
						results.add(temp);
				}	
				j++;
			}
		}
		return results;
	}

	/*
	 * Create calibration file. PRILM images are fitted against z-position based on angle between fitted centres of the two lobes of the psf.
	 */
	
	public static void calibrate(int inputPixelSize,int zStep)
	{		
		ImagePlus image 		= WindowManager.getCurrentImage(); 				// get current image, the calibration stack.
		int nFrames 			= image.getNFrames();							// number of frames in stack.
		if (nFrames == 1)														// different systems can store stacks in different ways, as frames or slices.
			nFrames 			= image.getNSlices(); 							// if stack stored as multislice image, use slice count instead of stack count.
		int nChannels 			= image.getNChannels();							// number of channels for calibration.
		int[] totalGain 		= {100,100,100,100,100,100,100,100,100,100}; 	// gain is not relevant for fitting, but fitting algorithms require this value to be sent.
		double meanRsquare 		= 0;											// used to determine if a new set of parameters yield a better calibration.
		int calibrationLength 	= 0;											// height in z obtained by the calibration.
		boolean[][] include 	= new boolean[6][10];							// for filtering of fitted results.
		double[][] lb 			= new double[6][10];							// for filtering of fitted results.
		double[][] ub 			= new double[6][10];							// for filtering of fitted results.
		for (int i = 0; i < 10; i++)											// populate "include", "lb" and "ub".
		{
			// logical list of which parameters to filter against.
			include[0][i] 		= false;
			include[1][i] 		= false;
			include[2][i] 		= true;		// r_square.
			include[3][i] 		= false;
			include[4][i] 		= false;
			include[5][i] 		= false;
			// lower bounds.
			lb[0][i]			= 0;
			lb[1][i]			= 0;
			lb[2][i]			= 0.8;		// r_square.
			lb[3][i]			= 0;
			lb[4][i]			= 0;
			lb[5][i]			= 0;
			// upper bounds.
			ub[0][i]			= 0;
			ub[1][i]			= 0;
			ub[2][i]			= 1.0;		// r_square.
			ub[3][i]			= 0;
			ub[4][i]			= 0;
			ub[5][i]			= 0;

		}

		double finalLevel 	= 0;			// final intensity used for creation of the calibration file.
		double finalSigma 	= 0;			// final max sigma used for creation of the calibration file.
		int finalDist 		= 0;			// final maximum distance between lobes.
		int gWindow 		= 5;			// initial window width for fitting.
		if (inputPixelSize < 100) 			// if smaller pixels then 100x100 nm, increase window width.
		{
			gWindow = (int) Math.ceil(500 / inputPixelSize); // 500 nm wide window.

			if (gWindow%2 == 0)			// gWindow needs to be odd.
				gWindow++;	
		}
		int finalGWindow = gWindow;		// final window for fitting.
		int loopC = 0;					// loop counter.
		/*
		 * Optimize variables.
		 */
		while (loopC < 2)				// loop through twice, each time with larger window width for fitting.
		{
			gWindow = gWindow + loopC*2; // increase window size each loop.
			for (double level = 0.7; level > 0.4; level -= 0.1)	// decrease intensity required for fitting.
			{
				for (double maxSigma = 2.5; maxSigma < 4; maxSigma += 0.5) // increase maximal sigma for fitting, lower value results in faster fitting.
				{
					for (int maxDist = 600; maxDist <= 1000; maxDist += 200) // increase maximum allowed distance between lobes.
					{
						int maxSqdist = maxDist * maxDist;
						int[] MinLevel 			= new int[10]; // precast.
						for (int ch = 1; ch <= nChannels; ch++) // set level in a channel by channel specific manner.
						{
							if (image.getNFrames() == 1)
							{
								image.setPosition(							
										ch,			// channel.
										(int) nFrames/2,			// slice.
										1);		// frame.
							}
							else
							{														
								image.setPosition(
										ch,			// channel.
										1,			// slice.
										(int) nFrames/2);		// frame.
							}
							ImageStatistics IMstat 	= image.getStatistics(); 					
							MinLevel[ch-1] = (int)(IMstat.max*level);
						}
						Fit3D.fit(MinLevel,gWindow,inputPixelSize,totalGain,maxSigma);	// fit data based on current parameters.
						/*
						 * clean out fits based on goodness of fit:
						 */

						cleanParticleList.run(lb,ub,include);			// remove all fits that fail to fullfill criteria.
						cleanParticleList.delete();
						ArrayList<Particle> result = TableIO.Load();						
						for (int i = 0; i < result.size(); i++) 		// loop over all entries.
						{
							result.get(i).z = (result.get(i).frame-1)*zStep; // set z to the current slice*slice step size.
						}
						TableIO.Store(result);
						result 				= TableIO.Load();
						int id 				= 2;		
						double[][] angle	= new double[nFrames][nChannels];	// angle for each slice and channel.
						int[][] count 	  	= new int[nFrames][nChannels];		// number of entries for this channel and z-slice.
						for (int Ch = 1; Ch <= nChannels; Ch++)					// loop over all channels.
						{
							for (int i = 0; i < result.size(); i++)				// loop over all entries
							{			
								if (result.get(i).include == 1 && result.get(i).channel == Ch) // if the current entry is within ok range and has not yet been assigned.
								{
									int idx = i + 1;	// start index for finding second lobe.
									while (idx < result.size() && result.get(i).channel == result.get(idx).channel)	// bounds and channel check.
									{
										if (result.get(i).frame == result.get(idx).frame && result.get(idx).include == 1)	// if within the same frame and not previously used in calculations.
										{
											if (((result.get(i).x - result.get(idx).x)*(result.get(i).x - result.get(idx).x) +					// check x
													(result.get(i).y - result.get(idx).y)*(result.get(i).y - result.get(idx).y)) < maxSqdist)	// and y distance is less the maxDistance.
											{
												result.get(idx).include = id;								// remove from further calculations.
												result.get(i).include 	= id;								// remove from further calculations.
												short dx = (short)(result.get(i).x - result.get(idx).x); 	// diff in x dimension.
												short dy = (short)(result.get(i).y - result.get(idx).y); 	// diff in y dimension.
												angle[(int)(result.get(i).frame-1)][Ch-1] += (Math.atan2(dy, dx)); // angle between points and horizontal axis.
												count[(int)(result.get(i).frame-1)][Ch-1]++;				// update counter for this slice and channel.
											}
										}
										idx++;
									}    				    				
									id++;
								}
							}
						}
						for (int Ch = 1; Ch <= nChannels; Ch++)	// for all channels.
						{
							for(int i = 0; i < count.length; i++)	// for all frames.
							{
								if (count[i][Ch-1]>0)				// if data has been entered.
								{
									angle[i][Ch-1] 	/= count[i][Ch-1]; // mean angle for this z-depth.					
								}
							}
						}
						int minLength = 40;															// minimum length of calibration range.		
						double[][] calibration = makeCalibrationCurve(angle,minLength,nChannels,false,false);// create calibration curve.
						if (calibrationLength < calibration.length)				// if the new calibration using current parameter settings covers a larger range.
						{
							calibrationLength = calibration.length;				// update best z range.
							meanRsquare = 0;									// set quailty check to 0.						
						}
						if (calibrationLength == calibration.length) 			// if of equal length (or if it was just updated).
						{
							double rsquare = 0;
							for (int i = 0; i < result.size(); i++)
							{
								rsquare +=result.get(i).r_square;				// calculate mean quality of fit for this parameter set.
							}
							rsquare /= result.size();
							if (rsquare > meanRsquare)							// if the new parameters yield better average fit.
							{								
								meanRsquare = rsquare;							// update.													
								finalLevel = level;								// update.						
								finalSigma = maxSigma;							// update.						
								finalDist = maxDist;							// update.						
								finalGWindow = gWindow;							// update.						
							}
						}
					} // iterate over maxDistance
				} // iterate over maxSigma
			} // iterate over level.
			loopC++;
		}

		/*
		 * Create calibration curve based on optimal parameters iterated over in the above code.
		 */
		int maxSqdist 			= finalDist * finalDist;		// final maximum distance between lobes.
		int[] MinLevel 			= new int[10]; // precast.
		for (int ch = 1; ch <= nChannels; ch++) // set level in a channel by channel specific manner.
		{
			if (image.getNFrames() == 1)
			{
				image.setPosition(							
						ch,			// channel.
						(int) nFrames/2,			// slice.
						1);		// frame.
			}
			else
			{														
				image.setPosition(
						ch,			// channel.
						1,			// slice.
						(int) nFrames/2);		// frame.
			}
			ImageStatistics IMstat 	= image.getStatistics(); 					
			MinLevel[ch-1] = (int)(IMstat.max*finalLevel);
		}
		Fit3D.fit(MinLevel,finalGWindow,inputPixelSize,totalGain,finalSigma); // fit all particles.
		
		/*
		 * clean out fits based on goodness of fit:
		 */

		cleanParticleList.run(lb,ub,include);
		cleanParticleList.delete();
		ArrayList<Particle> result = TableIO.Load();
		for (int i = 0; i < result.size(); i++)				// loop over all entries.
		{
			result.get(i).z = (result.get(i).frame-1)*zStep; // set z to zStep in nm * frame index.
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


		int minLength = 40;									// minimum length of calibration range.				
		double[][] calibration = makeCalibrationCurve(angle,minLength,nChannels,false,false);	// create calibration curve.
		/*
		 * STORE calibration file:
		 */

		ij.Prefs.set("SMLocalizer.calibration.PRILM.window",finalGWindow);
		ij.Prefs.set("SMLocalizer.calibration.PRILM.sigma",finalSigma);
		ij.Prefs.set("SMLocalizer.calibration.PRILM.distance",finalDist);

		ij.Prefs.set("SMLocalizer.calibration.PRILM.height",calibration.length);
		ij.Prefs.set("SMLocalizer.calibration.PRILM.channels",nChannels);
		ij.Prefs.set("SMLocalizer.calibration.PRILM.step",zStep);
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.PRILM.Ch"+Ch+"."+i,calibration[i][Ch-1]);
			}
		} 
		/*
		 * Pass result to fit function. Set PRILM.offset.x.ChId and PRILM.offset.y.ChId to 0 and determine this value through nearest neighbor between channels.
		 */
		for (int i = 2; i < 11; i++)
		{
			ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetX"+i,0);
			ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetY"+i,0);
			ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetZ"+i,0);
		}
		ij.Prefs.savePreferences(); // store settings.
		ArrayList<Particle> resultCalib = fit(result);
		TableIO.Store(resultCalib);
		/*
		 * Go through central part of the fit for each channel and calculate offset in XY for each channel.
		 */
		
		if (nChannels > 1)
		{
			int[] z = {zStep*(calibration.length/2 - 5),zStep*(calibration.length/2 + 5)}; // lower and upper bounds for z.
			double[][] offset = new double[3][nChannels-1];
			int[] counter = new int[nChannels-1];
			for (int i = 0; i < resultCalib.size(); i++)
			{
				if (resultCalib.get(i).channel == 1) // first channel.
				{
					if (resultCalib.get(i).z > z[0] && resultCalib.get(i).z < z[1]) // within center part of the calibration file.						
					{
						int ch = 2;
						while (ch <= nChannels) // reference all subsequent channels against the first one.
						{
							double particleDistance = inputPixelSize*inputPixelSize;
							int nearestNeighbor = 0;
							for (int j = i+1; j < resultCalib.size(); j++)
							{
								if (resultCalib.get(j).channel == ch)
								{
									if (resultCalib.get(i).x - resultCalib.get(j).x < inputPixelSize &&
											resultCalib.get(i).y - resultCalib.get(j).y < inputPixelSize)
									{
										// calculate distance between particle i and j.
										double tempDist = Math.sqrt((resultCalib.get(i).x - resultCalib.get(j).x)*(resultCalib.get(i).x - resultCalib.get(j).x) + 
												(resultCalib.get(i).y - resultCalib.get(j).y)*(resultCalib.get(i).y - resultCalib.get(j).y) + 
												(resultCalib.get(i).z - resultCalib.get(j).z)*(resultCalib.get(i).z - resultCalib.get(j).z) 
												);
										if(tempDist < particleDistance) 	// if j is closer then the previous 
										{
											nearestNeighbor = j;			// update which object is closest to i.
											particleDistance = tempDist;	// update the distance from i to j.
										}
									}
								}
							}
							counter[ch-2]++;
							offset[0][ch-2] += (resultCalib.get(nearestNeighbor).x - resultCalib.get(i).x); // remove this offset from channel ch.
							offset[1][ch-2] += (resultCalib.get(nearestNeighbor).y - resultCalib.get(i).y); // remove this offset from channel ch.
							offset[2][ch-2] += (resultCalib.get(nearestNeighbor).z - resultCalib.get(i).z); // remove this offset from channel ch.
							ch++;
						}
					}
				}
			}
			// normalize.
			for(int i = 0; i < nChannels-1; i ++)
			{
				offset[0][i] /=counter[i];
				offset[1][i] /=counter[i];
				offset[2][i] /=counter[i];
			}
			// store x y and z offsets for each channel.
			for (int i = 1; i < nChannels; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetX"+(i+1),offset[0][i-1]);
				ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetY"+(i+1),offset[1][i-1]);
				ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetZ"+(i+1),offset[2][i-1]);
			}
			ij.Prefs.savePreferences(); // store settings.
		}else
		{
			for (int i = 2; i < 11; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetX"+i,0);
				ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetY"+i,0);
				ij.Prefs.set("SMLocalizer.calibration.PRILM.ChOffsetZ"+i,0);
			}
			ij.Prefs.savePreferences(); // store settings.
		}
		
/*		double[] printout = new double[calibration.length];
		for (int i = 0; i < printout.length; i++)
			printout[i] = calibration[i][0];
		correctDrift.plot(printout);*/
	} // calibrate.
	public static double[][] getOffset()
	{
		double[][] offset = new double[3][(int) ij.Prefs.get("SMLocalizer.calibration.PRILM.channels",0)];
		for (int i = 2; i <= offset[0].length; i++)
		{
			offset[0][i-2] = ij.Prefs.get("SMLocalizer.calibration.PRILM.ChOffsetX"+i,0);
			offset[1][i-2] = ij.Prefs.get("SMLocalizer.calibration.PRILM.ChOffsetY"+i,0);
			offset[2][i-2] = ij.Prefs.get("SMLocalizer.calibration.PRILM.ChOffsetZ"+i,0);
		}
		
		return offset;
	}
	public static double getZ (double[][] calibration, int channel, double angle)
	{
		double z = 0;
		int idx = 1;		
		while (idx < calibration.length -1 && calibration[idx][channel-1] < angle)
		{
			idx++;
		}		
		if (idx == calibration.length -1 && angle > calibration[idx][channel-1])
			z = -1;
		else if (calibration[idx][channel-1] == angle)
			z = idx;
		else if (calibration[0][channel-1] == angle)
			z = 0;
		else if (calibration[0][channel-1] > angle)
			z = -1;
		else // interpolate
		{
			double diff = calibration[idx][channel-1] - calibration[idx - 1][channel-1];
			double fraction = (angle - calibration[idx - 1][channel-1]) / diff;
			z = idx - 1 + fraction;
		} 					
		if (z != -1)
			z *= ij.Prefs.get("SMLocalizer.calibration.PRILM.step",0); // scale.
		return z;
	}

	// returns calibration[zStep][channel]
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

	/*
	 * Get calibration curve through 5 point moving mean.
	 */

	public static double[][] makeCalibrationCurve(double[][] result, int minLength, int nChannels,boolean printout, boolean full)
	{
		int[] start 	= new int[nChannels]; 	// start index for calibration, channel specific.
		int[] end 		= new int[nChannels]; 	// end index for calibration, channel specific.
		int maxCounter 	= 0;
		int channelIdx 	= 1;					// current channel.

		while (channelIdx <= nChannels)			// loop over all channels.
		{
			double[] tempVector = new double[result.length];			// 5 point smoothed data.
			int idx 					= 0;							// index variable.
			boolean iterate 			= true;							// loop variable.

			while (idx < result.length)
			{
				// 5 point smoothing with edge handling.
				if (idx == 0)
				{
					int included = 0;
					if (result[idx][channelIdx-1] != 0)
						included++;
					if (result[idx + 1][channelIdx-1] != 0)
						included++;
					if (result[idx + 2][channelIdx-1] != 0)
						included++;
					if (included > 0)
					{
						tempVector[idx] = (result[idx][channelIdx-1]
								+ result[idx + 1][channelIdx-1] 
										+ result[idx + 2][channelIdx-1])/included;
					}
					else
					{
						tempVector[idx] = 0;
					}
				}else if (idx == 1)
				{
					int included = 0;					
					if (result[idx - 1][channelIdx-1] != 0)
						included++;
					if (result[idx][channelIdx-1] != 0)
						included++;
					if (result[idx + 1][channelIdx-1] != 0)
						included++;
					if (result[idx + 2][channelIdx-1] != 0)
						included++;
					if (included > 0)
					{
						tempVector[idx] = (result[idx - 1][channelIdx-1]
								+ result[idx][channelIdx-1]
										+ result[idx + 1][channelIdx-1] 
												+ result[idx + 2][channelIdx-1])/included;
					}
					else
					{
						tempVector[idx] = 0;
					}
				}else if (idx == result.length - 2)
				{
					int included = 0;
					if (result[idx - 2][channelIdx-1] != 0)
						included++;
					if (result[idx - 1][channelIdx-1] != 0)
						included++;
					if (result[idx][channelIdx-1] != 0)
						included++;
					if (result[idx + 1][channelIdx-1] != 0)
						included++;
					if (included > 0)
					{
						tempVector[idx] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1] 
												+ result[idx + 1][channelIdx-1])/included;
					}
					else
					{
						tempVector[idx] = 0;
					}
				}else if (idx == result.length - 1)
				{
					int included = 0;
					if (result[idx - 2][channelIdx-1] != 0)
						included++;
					if (result[idx - 1][channelIdx-1] != 0)
						included++;
					if (result[idx][channelIdx-1] != 0)
						included++;
					if (included > 0)
					{
						tempVector[idx] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1])/included;
					}
					else
					{
						tempVector[idx] = 0;
					}
				}else if (idx < result.length - 2)
				{
					int included = 0;
					if (result[idx - 2][channelIdx-1] != 0)
						included++;
					if (result[idx - 1][channelIdx-1] != 0)
						included++;
					if (result[idx][channelIdx-1] != 0)
						included++;
					if (result[idx + 1][channelIdx-1] != 0)
						included++;
					if (result[idx + 2][channelIdx-1] != 0)
						included++;
					if (included > 0)
					{
						tempVector[idx] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1] 
												+ result[idx + 1][channelIdx-1]
														+ result[idx + 2][channelIdx-1])/included;
					}
					else
					{
						tempVector[idx] = 0;
					}
				}
				idx++;

			} // tempVector populated.

			// find first and last position for both result and selectionParameter vectors.

			idx 		= 1;		// reset.
			int counter = 0;		// counter to keep track of how long the calibration curve is continuous for.
			while (idx < result.length && iterate)
			{
				if (tempVector[idx] >= 0)
				{

					if (counter > minLength)
					{
						end[channelIdx-1] = idx - 1;
						iterate = false;
					}
					counter = 0; // reset
				}

				else
				{
					if (tempVector[idx] > tempVector[idx - 1]) 
						counter++;
					if (tempVector[idx] < tempVector[idx - 1])
					{
						if (counter > minLength)
						{
							end[channelIdx-1] = idx - 1;
							iterate = false;

						}
						else
							counter = 0;
					}
				}
				if (counter == minLength)
				{
					start[channelIdx-1] = idx - minLength + 1;
				}

				idx++;
				if (idx == result.length && iterate)
				{
					if (counter > minLength)
						end[channelIdx-1] = idx-1;
				}
			}
			if (start[channelIdx-1] == 1)
			{
				if (tempVector[0] < tempVector[1])
					start[channelIdx-1] = 0;
			}
			channelIdx++;
		}


		if(printout)
		{				
			channelIdx--;
			System.out.println("start: " + start[channelIdx-1] + " end: " + end[channelIdx-1] + " length: " + (end[channelIdx-1] - start[channelIdx-1] + 1));
		}

		for (int i = 0; i < nChannels; i++)
		{
			if (maxCounter < end[i] - start[i] + 1)
			{
				maxCounter = end[i] - start[i] + 1;
			}
		}
		if (full)	// if true, set start and end to cover full range.
		{
			end[0] = result.length-1;
			start[0] = 0;
			maxCounter = end[0] - start[0] + 1;	
		}
		if (maxCounter < 0)	// error check.
			maxCounter = 0;
		if (maxCounter >= minLength)
		{
			double[][] calibration = new double[maxCounter][nChannels];	// output matrix.
			channelIdx = 1;												// reset 

			while (channelIdx <= nChannels)					// loop over all channels.
			{
				int idx = start[channelIdx-1];
				int count = 0;
				while (idx <= end[channelIdx-1])				
				{
					// 5 point smoothing:
					if (idx == start[channelIdx-1])
					{
						int included = 0;
						if (result[idx][channelIdx-1] != 0)
							included++;
						if (result[idx + 1][channelIdx-1] != 0)
							included++;
						if (result[idx + 2][channelIdx-1] != 0)
							included++;
						if (included > 0)
							calibration[count][channelIdx-1] = (result[idx][channelIdx-1]
									+ result[idx + 1][channelIdx-1] 
											+ result[idx + 2][channelIdx-1])/included;
						else
							calibration[count][channelIdx-1] = 0;
					}else if (idx == start[channelIdx-1] + 1)
					{
						int included = 0;					
						if (result[idx - 1][channelIdx-1] != 0)
							included++;
						if (result[idx][channelIdx-1] != 0)
							included++;
						if (result[idx + 1][channelIdx-1] != 0)
							included++;
						if (result[idx + 2][channelIdx-1] != 0)
							included++;
						if (included > 0)
							calibration[count][channelIdx-1] = (result[idx - 1][channelIdx-1]
									+ result[idx][channelIdx-1]
											+ result[idx + 1][channelIdx-1] 
													+ result[idx + 2][channelIdx-1])/included;
						else
							calibration[count][channelIdx-1] = 0;
					}else if (idx == end[channelIdx-1] - 1)
					{
						int included = 0;
						if (result[idx - 2][channelIdx-1] != 0)
							included++;
						if (result[idx - 1][channelIdx-1] != 0)
							included++;
						if (result[idx][channelIdx-1] != 0)
							included++;
						if (result[idx + 1][channelIdx-1] != 0)
							included++;
						if (included > 0)
							calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
									+ result[idx - 1][channelIdx-1]
											+ result[idx][channelIdx-1] 
													+ result[idx + 1][channelIdx-1])/included;
						else
							calibration[count][channelIdx-1] = 0;
					}else if (idx == end[channelIdx-1])
					{
						int included = 0;
						if (result[idx - 2][channelIdx-1] != 0)
							included++;
						if (result[idx - 1][channelIdx-1] != 0)
							included++;
						if (result[idx][channelIdx-1] != 0)
							included++;
						if (included > 0)
							calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
									+ result[idx - 1][channelIdx-1]
											+ result[idx][channelIdx-1])/included;
						else
							calibration[count][channelIdx-1] = 0;
					}else if (idx < end[channelIdx-1] - 1)
					{
						int included = 0;
						if (result[idx - 2][channelIdx-1] != 0)
							included++;
						if (result[idx - 1][channelIdx-1] != 0)
							included++;
						if (result[idx][channelIdx-1] != 0)
							included++;
						if (result[idx + 1][channelIdx-1] != 0)
							included++;
						if (result[idx + 2][channelIdx-1] != 0)
							included++;
						if (included > 0)
							calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
									+ result[idx - 1][channelIdx-1]
											+ result[idx][channelIdx-1] 
													+ result[idx + 1][channelIdx-1]
															+ result[idx + 2][channelIdx-1])/included;
						else
							calibration[count][channelIdx-1] = 0;
					}
					count++;
					idx++;
				}
				channelIdx++;
			}
			return calibration;
		}

		else	// if calibration failed.
		{
			double[][] calibration = new double[maxCounter][nChannels];
			return calibration;
		}
	}
}