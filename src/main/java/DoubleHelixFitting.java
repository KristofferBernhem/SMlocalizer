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
import java.awt.Color;
import java.util.ArrayList;

import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.Plot;
import ij.plugin.filter.Analyzer;
import ij.process.ImageStatistics;

public class DoubleHelixFitting {
	public static ArrayList<Particle> fit(ArrayList<Particle> inputResults)
	{
		ArrayList<Particle> results = new ArrayList<Particle>();
		double[][] calibration 		= getCalibration();
		int distance 				= (int)ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.distance",0);
		distance *= distance; // square max distance between centers.
		double[][] offset = getOffset();
		double[] zOffset  = getZoffset();     // get where z=0 is for each channel in the calibration file.

		for (int i = 0; i < inputResults.size(); i++)
		{
			inputResults.get(i).include = 2;
			int j = i+1;
			int closest = 0;
			double NNdistance = distance;
			while (j < inputResults.size() && inputResults.get(i).frame == inputResults.get(j).frame
					&& inputResults.get(i).channel == inputResults.get(j).channel)
			{
				if (inputResults.get(j).include == 1)
				{
					double currentDistance =(inputResults.get(i).x - inputResults.get(j).x)*(inputResults.get(i).x - inputResults.get(j).x) + 
							(inputResults.get(i).y - inputResults.get(j).y)*(inputResults.get(i).y - inputResults.get(j).y); 
					if (currentDistance < NNdistance)
					{
						closest = j;
						NNdistance = currentDistance;
					}
				}
				j++;
			}
			if (closest != 0)
				j = closest;
			if (closest != 0 )//&&
//					inputResults.get(i).photons < 1.2*inputResults.get(j).photons &&
//					inputResults.get(i).photons > 0.8*inputResults.get(j).photons )
			{	
				
				{
					inputResults.get(j).include = 2;
					short dx 		= (short)(inputResults.get(i).x - inputResults.get(j).x); // diff in x dimension.
					short dy 		= (short)(inputResults.get(i).y - inputResults.get(j).y); // diff in y dimension.
					double angle 	= (Math.atan2(dy, dx));
					
					// find new points shifted by the precision of radial localization and use the change in angle between these two new points and the orignal ones to calculate the error in z determination.
					short xiprime = (short) (inputResults.get(i).precision_x * Math.cos(angle+Math.PI*0.5) + inputResults.get(i).x); // shift in x along the normal to the vector connecting the two lobes.
					short yiprime = (short) (inputResults.get(i).precision_y * Math.sin(angle+Math.PI*0.5) + inputResults.get(i).y); // shift in y along the normal to the vector connecting the two lobes.
					short xjprime = (short) (inputResults.get(j).precision_x * Math.cos(angle-Math.PI*0.5) + inputResults.get(j).x); // shift in x along the normal to the vector connecting the two lobes.
					short yjprime = (short) (inputResults.get(j).precision_y * Math.sin(angle-Math.PI*0.5) + inputResults.get(j).y); // shift in y along the normal to the vector connecting the two lobes.
					
					double angleError = angle - Math.atan2((yiprime - yjprime),(xiprime - xjprime));			// symetry allows us to calculate a single angle.This is the propagated error in angle used in depth calibration.
					double zLow = getZ(calibration, zOffset, inputResults.get(i).channel, angle - angleError); // lower bound.
					double zHigh = getZ(calibration, zOffset, inputResults.get(i).channel, angle + angleError); // upper bound.
					// translate angle to z:
					Particle temp 	= new Particle();
					temp.channel 	= inputResults.get(i).channel;
					temp.z 		 	= getZ(calibration, zOffset, temp.channel, angle);
					temp.frame 	 	= inputResults.get(i).frame;
					temp.photons 	= inputResults.get(i).photons + inputResults.get(j).photons;
					temp.x			= (inputResults.get(i).x + inputResults.get(j).x) / 2;
					temp.y			= (inputResults.get(i).y + inputResults.get(j).y) / 2;
					temp.sigma_x 	= (inputResults.get(i).sigma_x + inputResults.get(j).sigma_x) / 2; 				// fitted sigma in x direction.
					temp.sigma_y 	= (inputResults.get(i).sigma_y + inputResults.get(j).sigma_y) / 2; 				// fitted sigma in y direction.
					temp.precision_x= 0.5*(inputResults.get(i).precision_x+inputResults.get(j).precision_x); 	// precision of fit for x coordinate.
					temp.precision_y= 0.5*(inputResults.get(i).precision_y+inputResults.get(j).precision_y); 	// precision of fit for y coordinate.
					temp.precision_z= 0.5*((temp.z - zLow)+(zHigh-temp.z)); 									// precision of fit for z coordinate.
					temp.r_square 	= Math.min(inputResults.get(i).r_square,inputResults.get(j).r_square);; 		// Goodness of fit.
					temp.include	= 1; 																			// If this particle should be included in analysis and plotted.
					if (zLow == 1E6 || zHigh == 1E6)
						temp.precision_z = 1000; // if outside of the calibration range, set to 1000.

					if(temp.z != 1E6 && temp.channel>1)// if within ok z range. For all but first channel, move all particles by x,y,z offset for that channel to align all to first channel.
					{
						temp.x -= offset[0][temp.channel-1];
						temp.y -= offset[1][temp.channel-1];
						temp.z -= offset[2][temp.channel-1];
						results.add(temp);
					}

					if (temp.z != 1E6 && temp.channel==1) // if within ok z range. Don't shift first channel.
						results.add(temp);
				}
			}


		}
		results = shiftXY(results);
		return results;
	}
	/*
	 * Create calibration file. Double helix images are fitted against z-position based on angle between the two lobes of the psf.  
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
		boolean[][] include 	= new boolean[7][10];							// for filtering of fitted results.
		double[][] lb 			= new double[7][10];							// for filtering of fitted results.
		double[][] ub 			= new double[7][10];							// for filtering of fitted results.
		for (int i = 0; i < 10; i++)											// populate "include", "lb" and "ub".
		{
			// logical list of which parameters to filter against.
			include[0][i] 		= false;
			include[1][i] 		= false;
			include[2][i] 		= true;		// r_square.
			include[3][i] 		= false;
			include[4][i] 		= false;
			include[5][i] 		= false;
			include[6][i] 		= false;
			// lower bounds.
			lb[0][i]			= 0;
			lb[1][i]			= 0;
			lb[2][i]			= 0.8;		// r_square.
			lb[3][i]			= 0;
			lb[4][i]			= 0;
			lb[5][i]			= 0;
			lb[6][i]			= 0;
			// upper bounds.
			ub[0][i]			= 0;
			ub[1][i]			= 0;
			ub[2][i]			= 1.0;		// r_square.
			ub[3][i]			= 0;
			ub[4][i]			= 0;
			ub[5][i]			= 0;
			ub[6][i]			= 0;

		}

		double finalLevel 	= 0;			// final intensity used for creation of the calibration file.
		double finalSigma 	= 0;			// final max sigma used for creation of the calibration file.
		int finalDist 		= 0;			// final maximum distance between lobes.
		int gWindow 		= 5;			// initial window width for fitting.
		if (inputPixelSize < 100) 			// if smaller pixels then 100x100 nm, increase window width.
		{
			gWindow = (int) Math.ceil(500 / inputPixelSize); // 500 nm wide window.

			if (gWindow%2 == 0)				// gWindow needs to be odd.
				gWindow++;	
		}
		int finalGWindow	= gWindow;		// final window for fitting.
		int loopC 			= 0;			// loop counter.
		/*
		 * Optimize variables.
		 */
		while (loopC < 2)				// loop through twice, each time with larger window width for fitting.
		{
			gWindow = gWindow + loopC*2; // increase window size each loop.
			for (double level = 0.7; level > 0.4; level -= 0.1)
			{
				for (double maxSigma = 2.5; maxSigma < 4; maxSigma += 0.5)
				{
					for (int maxDist = 600; maxDist <= 1200; maxDist += 200)
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
						double[][] angle	= new double[nFrames][nChannels];	// angle between lobes in each frame and channel.
						double[][] distance = new double[nFrames][nChannels];	// distance between lobes in each frame and channel.
						int[][] count 	  	= new int[nFrames][nChannels];		// number of objects in each frame and channel.
						for (int Ch = 1; Ch <= nChannels; Ch++)
						{
							for (int i = 0; i < result.size(); i++)
							{			
								double NNdistance = maxSqdist;
								if (result.get(i).include == 1 && result.get(i).channel == Ch) // if the current entry is within ok range and has not yet been assigned.
								{
									int idx = i + 1;
									int j = 0;
									while (idx < result.size() && result.get(i).channel == result.get(idx).channel)
									{
										if (result.get(i).frame == result.get(idx).frame && result.get(idx).include == 1)
										{
											double currentDistance = (result.get(i).x - result.get(idx).x)*(result.get(i).x - result.get(idx).x) +
													(result.get(i).y - result.get(idx).y)*(result.get(i).y - result.get(idx).y);
											if (currentDistance < NNdistance)
											{
												j = idx;
											}
										}
										idx ++;
									}


									if (j != 0)
										idx = j;

									if (j != 0)
									{									
										result.get(idx).include = id;
										result.get(i).include 	= id;								
										short dx = (short)(result.get(i).x - result.get(idx).x); // diff in x dimension.
										short dy = (short)(result.get(i).y - result.get(idx).y); // diff in y dimension.
										angle[(int)(result.get(i).frame-1)][Ch-1] += (Math.atan2(dy, dx)); // angle between points and horizontal axis.
										if (Math.sqrt(dx*dx + dy*dy) > distance[(int)(result.get(i).frame-1)][Ch-1])
											distance[(int)(result.get(i).frame-1)][Ch-1] = Math.sqrt(dx*dx + dy*dy);
										count[(int)(result.get(i).frame-1)][Ch-1]++;												
										id++;
									}
								}
							}
						}
						for (int Ch = 1; Ch <= nChannels; Ch++) // for all channels.
						{
							for(int i = 0; i < count.length; i++) // for all frames.
							{
								if (count[i][Ch-1]>0)				// if data has been entered.
								{
									angle[i][Ch-1] 	/= count[i][Ch-1]; // mean angle for this z-depth.			
								}
							}
						}
						int minLength = 40;															// minimum length of calibration range.			
						double[][] calibration = makeCalibrationCurve(angle,minLength,nChannels,false,false,false);// create calibration curve.
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
								maxDist = 0;
								for (int ch = 0; ch < nChannels; ch++)
								{
									for (int i = 0; i < nFrames; i++)
									{
										if (distance[i][ch] > maxDist)
											maxDist = (int) distance[i][ch];
									}
								}
								finalDist = maxDist;							// update.						
								finalGWindow = gWindow;							// update.	
							}
						}
					} // iterate over maxDistance
				} // iterate over maxSigma
			} // iterate over level.
			loopC++;
		}

		int maxSqdist 			= finalDist * finalDist;
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
		Fit3D.fit(MinLevel,finalGWindow,inputPixelSize,totalGain,finalSigma);

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
		//		double[][][] xyOffset	= new double[2][nFrames][nChannels]; // xy(z) offset. xy(0) = 0;
		double[][] distance = new double[nFrames][nChannels];
		int[][] count 	  	= new int[nFrames][nChannels];
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{

			for (int i = 0; i < result.size(); i++)
			{		
				double NNdistance = maxSqdist;
				if (result.get(i).include == 1 && result.get(i).channel == Ch) // if the current entry is within ok range and has not yet been assigned.
				{
					int idx = i + 1;
					int j = 0;
					while (idx < result.size() && result.get(i).channel == result.get(idx).channel)
					{
						if (result.get(i).frame == result.get(idx).frame && result.get(idx).include == 1)
						{
							double currentDistance = (result.get(i).x - result.get(idx).x)*(result.get(i).x - result.get(idx).x) +
									(result.get(i).y - result.get(idx).y)*(result.get(i).y - result.get(idx).y);
							if (currentDistance < NNdistance)
							{
								j = idx;
							}
						}
						idx ++;
					}


					if (j != 0)
						idx = j;

					if (j != 0)
					{									
						result.get(idx).include = id;
						result.get(i).include 	= id;								
						short dx = (short)(result.get(i).x - result.get(idx).x); // diff in x dimension.
						short dy = (short)(result.get(i).y - result.get(idx).y); // diff in y dimension.
						angle[(int)(result.get(i).frame-1)][Ch-1] += (Math.atan2(dy, dx)); // angle between points and horizontal axis.
						if (Math.sqrt(dx*dx + dy*dy) > distance[(int)(result.get(i).frame-1)][Ch-1])
							distance[(int)(result.get(i).frame-1)][Ch-1] = Math.sqrt(dx*dx + dy*dy);
						count[(int)(result.get(i).frame-1)][Ch-1]++;												
						id++;
					}
				}
			}
		}
		for (int Ch = 1; Ch <= nChannels; Ch++)			// for all channels.
		{
			for(int i = 0; i < count.length; i++)		// for all frames.
			{
				if (count[i][Ch-1]>0)					// if particles has been included in this frame.
				{
					angle[i][Ch-1] 	/= count[i][Ch-1]; // mean angle for this z-depth.
				}
			}
		}
		int minLength = 40;			
		double[][] calibration = makeCalibrationCurve(angle,minLength,nChannels,false,false,true);

		/*
		 * STORE calibration file:
		 */

		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.window",finalGWindow);
		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.sigma",finalSigma);
		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.distance",finalDist);

		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.height",calibration.length);
		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.channels",nChannels);
		ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.step",zStep);
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.Ch"+Ch+"."+i,calibration[i][Ch-1]);
			}
		} 
		for (int i = 2; i < 11; i++)
		{
			ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetX"+i,0);
			ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetY"+i,0);
			ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetZ"+i,0);
		}
		for (int ch = 1; ch <= nChannels; ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.xOffset.Ch"+ch+"."+i,0);  // reset.
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.yOffset.Ch"+ch+"."+i,0);	// reset.
			}
		}
		ij.Prefs.savePreferences(); // store settings.
		for (int idx = 0; idx < result.size(); idx++)
		{
			result.get(idx).include = 1;
		}
		ArrayList<Particle> resultCalib = fit(result);
		TableIO.Store(resultCalib);
		cleanParticleList.run(lb,ub,include);
		cleanParticleList.delete();
		resultCalib = TableIO.Load();
		double[][][] xyOffset = getXYoffset(resultCalib,calibration,nFrames);
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();
		tab.reset();
		tab.show("Results");
		/*
		 * Go through central part of the fit for each channel and calculate offset in XY for each channel.
		 */

		for (int ch = 1; ch <= nChannels; ch++)
		{
			for (int i = 0; i < calibration.length; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.xOffset.Ch"+ch+"."+i,xyOffset[0][i][ch-1]);
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.yOffset.Ch"+ch+"."+i,xyOffset[1][i][ch-1]);
			}
		}
		ij.Prefs.savePreferences(); // store settings.

		resultCalib = shiftXY(resultCalib);
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
										double tempDist = Math.sqrt((resultCalib.get(i).x - resultCalib.get(j).x)*(resultCalib.get(i).x - resultCalib.get(j).x) + 
												(resultCalib.get(i).y - resultCalib.get(j).y)*(resultCalib.get(i).y - resultCalib.get(j).y) + 
												(resultCalib.get(i).z - resultCalib.get(j).z)*(resultCalib.get(i).z - resultCalib.get(j).z) 
												);
										if(tempDist < particleDistance)
										{
											nearestNeighbor = j;
											particleDistance = tempDist;
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
			for(int i = 0; i < nChannels-1; i ++)
			{
				offset[0][i] /=counter[i];
				offset[1][i] /=counter[i];
				offset[2][i] /=counter[i];
			}
			for (int i = 1; i < nChannels; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetX"+(i+1),offset[0][i-1]);
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetY"+(i+1),offset[1][i-1]);
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetZ"+(i+1),offset[2][i-1]);
			}
			ij.Prefs.savePreferences(); // store settings.
		}else
		{
			for (int i = 2; i < 11; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetX"+i,0);
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetY"+i,0);
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.ChOffsetZ"+i,0);
			}
			ij.Prefs.savePreferences(); // store settings.
		}

		Plot plot = new Plot("Double helix calibration", "z [nm]", "Angle [rad]");
		double[] zOffset = getZoffset();
		for (int ch = 0; ch < calibration[0].length; ch++)
		{
			double[] printout = new double[calibration.length];
			double[] x = new double[printout.length];
			for (int i = 0; i < printout.length; i++)
			{
				printout[i] = calibration[i][ch];
				x[i] = (i-zOffset[ch])*zStep;
			}
			if (ch == 0)
				plot.setColor(Color.BLACK);
			if (ch == 1)
				plot.setColor(Color.BLUE);
			if (ch == 2)
				plot.setColor(Color.RED);
			if (ch == 3)
				plot.setColor(Color.GREEN);
			if (ch == 4)
				plot.setColor(Color.MAGENTA);
			plot.addPoints(x,printout, Plot.LINE);
		}
		if (calibration[0].length == 1)
			plot.addLegend("Ch 1 \n Ch 2");
		if (calibration[0].length == 2)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3");		
		if (calibration[0].length == 3)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3 \n Ch 4");
		if (calibration[0].length == 4)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3 \n Ch 4 \n Ch 5");
		plot.show();


	} // calibrate.

	public static double[][] getOffset()
	{
		double[][] offset = new double[3][(int) ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.channels",0)];
		for (int i = 2; i <= offset[0].length; i++)
		{
			offset[0][i-2] = ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.ChOffsetX"+i,0);
			offset[1][i-2] = ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.ChOffsetY"+i,0);
			offset[2][i-2] = ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.ChOffsetZ"+i,0);
		}

		return offset;
	}
	public static double getZ (double[][] calibration, double[] zOffset, int channel, double angle)
	{
		double z = 0;
		int idx = 1;
		while (idx < calibration.length -1 && calibration[idx][channel-1] > angle)
		{
			idx++;
		}
		if (idx == calibration.length -1 && angle < calibration[idx][channel-1])
			z = 1E6;
		else if (calibration[idx][channel-1] == angle)
			z = idx;
		else if (calibration[0][channel-1] == angle)
			z = 0;
		else if (calibration[0][channel-1] < angle)
			z = 1E6;					
		else // interpolate
		{
			double diff = calibration[idx][channel-1] - calibration[idx - 1][channel-1];
			double fraction = (angle - calibration[idx - 1][channel-1]) / diff;
			z = idx - 1 + fraction;
		} 	

		if (z != 1E6)
		{
			z -= zOffset[channel-1];
			z *= ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.step",0); // scale.
		}
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
	public static ArrayList<Particle> shiftXY(ArrayList<Particle> inputList)
	{
		double[][][] xyOffset = new double[2][(int) ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.height",0)][(int) ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.channels",0)]; // precast.
		for(int ch = 1; ch <= xyOffset[0][0].length; ch++) // load in correction table.
		{
			for (int i = 0; i < xyOffset[0].length; i++)	// loop over all z positions.

			{
				xyOffset[0][i][ch-1] =  ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.xOffset.Ch"+ch+"."+i,0);
				xyOffset[1][i][ch-1] =  ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.yOffset.Ch"+ch+"."+i,0);
			}
		}	
		double zStep =  ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.step",0);
		double[] center = getZoffset();
		for (int idx = 0; idx < inputList.size(); idx++)		 // loop over all particles.
		{
			double z = inputList.get(idx).z /zStep;				// normalize z position based on calibration z-step size.
			z += center[inputList.get(idx).channel-1];			// shift z by center to get back approximate frame number from stack. Use this to know what value to use from the offset table.
			double shiftX = 0;
			double shiftY = 0;
			if (z >= 0 && z < xyOffset[0].length)				// if within ok range.
			{
				shiftX = xyOffset[0][(int)Math.floor(z)][inputList.get(idx).channel-1];
				shiftX += (xyOffset[0][(int)(Math.floor(z)+1)][inputList.get(idx).channel-1] -xyOffset[0][(int)z][inputList.get(idx).channel-1]) * (Math.floor(z+1)-(int)(Math.floor(z)));
				shiftY = xyOffset[1][(int)Math.floor(z)][inputList.get(idx).channel-1];
				shiftY += (xyOffset[1][(int)(Math.floor(z)+1)][inputList.get(idx).channel-1] -xyOffset[1][(int)z][inputList.get(idx).channel-1]) * (Math.floor(z+1)-(int)(Math.floor(z)));
			}else if ((int)z == xyOffset[0].length) // special case if we're at the end of the table.
			{
				shiftX = xyOffset[0][(int)Math.floor(z)][inputList.get(idx).channel-1];
				shiftY = xyOffset[1][(int)Math.floor(z)][inputList.get(idx).channel-1];			
			}
			inputList.get(idx).x +=shiftX;	// add the compensation to x.
			inputList.get(idx).y +=shiftY;  // add the compensation to y.
		}

		return inputList;	// return corrected result.
	}
	public static double[] getZoffset()
	{
		int nChannels = (int)ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.channels",0);
		double[] zOffset = new double[nChannels];
		for (int Ch = 1; Ch <= nChannels; Ch++)
		{		
			zOffset[Ch-1] = ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.center.Ch"+Ch,0);
		} 	
		return zOffset;
	}
	public static double[][][] getXYoffset(ArrayList<Particle> inputParticle,double[][] calibration, int nFrames)
	{
		double[][][] xyOffset 	= new double[2][calibration.length][calibration[0].length]; // x-y (z,ch). 
		double[] center 		= getZoffset();			// get info of where z=0 is in the calibration file.
		nFrames /= 2; // center frame.	
		for (int ch = 0; ch < calibration[0].length; ch++)	// over all channels.
		{
			int chStart 	= -1;
			int nCenter 	= 0; // number of locations to search after.
			for (int idx 	= 0; idx < inputParticle.size(); idx++)
			{				
				if (inputParticle.get(idx).channel == ch+1 && inputParticle.get(idx).frame >= (nFrames - center[ch]) && chStart == -1)
					chStart = idx;
				if (inputParticle.get(idx).frame == nFrames && inputParticle.get(idx).channel == ch+1) // if we're at the frame corresponding to z = 0 and in the correct channel.
					//					if (inputParticle.get(idx).frame == center[ch] && inputParticle.get(idx).channel == ch+1) // if we're at the frame corresponding to z = 0 and in the correct channel.
				{
					nCenter++;
				}
			}
			if (nCenter > 0)	// if we found centers.
			{				
				double[][] localZeroXY = new double[2][nCenter];	// array to store all individual offsets.
				int counter  = 0;	
				for (int idx = 0; idx < inputParticle.size(); idx++)
				{
					if (inputParticle.get(idx).frame == nFrames && inputParticle.get(idx).channel == ch+1)	// if we're at the center frame.
						//						if (inputParticle.get(idx).frame == center[ch] && inputParticle.get(idx).channel == ch+1)	// if we're at the center frame.
					{
						localZeroXY[0][counter] = inputParticle.get(idx).x;	// store x coordinate.
						localZeroXY[1][counter] = inputParticle.get(idx).y;	// store y coordinate.
						counter++;
					}
				}// localZeroXY now contains x-y coordinates for central slice particles.
				/*
				 * Find smallest offset against localZeroXY for each particle in stack for this channel. Take mean, remove outliers.
				 */
				int startIdx = chStart; // which inded to start from.
				while (inputParticle.get(startIdx).channel != ch+1) // jump forward to first index for the current channel,
					startIdx ++;
				int endIdx		 = startIdx; // final index for this frame and channel.
				boolean optimize = true;	// loop whilst this is true.

				while (optimize)
				{
					counter = 0;	// keep track of number of added corrections.
					endIdx = startIdx;	// restart.
					boolean loop = true;	// exit flag for final frame and channel.
					if (inputParticle.size() > endIdx+1) // if we're not at the end of the list.
					{
						while (inputParticle.get(startIdx).frame == inputParticle.get(endIdx).frame && inputParticle.get(endIdx).channel == ch+1 && loop) // as long as we're within the same frame and channel.
						{
							if (inputParticle.size() > endIdx+2) // if we're not at the end
							{
								endIdx++;						 // step forward

							}else
								loop = false; // exit for final frame and channel.
						}
					}
					double[][] frameOffset = new double[2][endIdx-startIdx + 1]; // preallocate
					double meanX = 0;	// mean x offset.
					double meanY = 0;	// mean y offset.
					while (startIdx <= endIdx) // whilst we're not done with this frame.
					{
						double z = inputParticle.get(startIdx).z / ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.step",0);  // get back approximate frame.
						//	z += center[ch];																						// shift by center to get frame number.
						z += nFrames;																						// shift by center to get frame number.
						//			System.out.println(inputParticle.get(startIdx).frame + " vs  " + z   ); 
						if (z > (inputParticle.get(startIdx).frame - 4) && z < (inputParticle.get(startIdx).frame + 4)) 		// if we're within +/- 3 frames of the current one (ie if the fit worked with 3*zStep precision)
						{
							double distance = 500; // max distance to be considered in nm.
							for (int i = 0; i < localZeroXY[0].length; i ++) // loop over all particles from center frame.
							{
								double tempDistance = Math.sqrt((inputParticle.get(startIdx).x - localZeroXY[0][i])*(inputParticle.get(startIdx).x - localZeroXY[0][i]) + // calculate distance from current particle to the current center frame particle.
										(inputParticle.get(startIdx).y - localZeroXY[1][i])*(inputParticle.get(startIdx).y - localZeroXY[1][i]));
								if (tempDistance < distance) // if the particles are closer then the previous loops (or start value)
								{
									frameOffset[0][counter] = localZeroXY[0][i] - inputParticle.get(startIdx).x;	// update offset for this particle
									frameOffset[1][counter] = localZeroXY[1][i] - inputParticle.get(startIdx).y;	// update offset for this particle
									distance = tempDistance;														// update distance.
								}
							} // nearest neighbor found.
							meanX += frameOffset[0][counter]; 	// add offset in x to mean.
							meanY += frameOffset[1][counter]; 	// add offset in y to mean.
							counter ++;						 	// step forward in the array
						}
						startIdx++;								// next particle within the frame.
					}
					double sigmaX = 0;	// std of x offsets.
					double sigmaY = 0; 	// std of y offsets.
					if (counter > 0)	// if we've added values (ie found particles close enough to the center frame particles)
					{
						if (counter > 1) // sample requires more then 1 entry.
						{
							for (int i = 0; i < frameOffset[0].length; i++)	// loop over all offsets in this frame.
							{
								sigmaX += ((frameOffset[0][i] - meanX/counter)*(frameOffset[0][i] - meanX/counter))/(counter-1);	// add (Xi-µ)/(n-1)
								sigmaY += ((frameOffset[1][i] - meanY/counter)*(frameOffset[1][i] - meanY/counter))/(counter-1);	// add (Yi-µ)/(n-1)
							}
							sigmaX = Math.sqrt(sigmaX);	// square root of sigmax^2.
							sigmaY = Math.sqrt(sigmaY);	// square root of sigmay^2.	
							//			System.out.println("frame: " + inputParticle.get(endIdx).frame + " mean" + meanX/counter + " x " + meanY/counter + " sigma: " + sigmaX + " x " + sigmaY);
							for (int i = 0; i < frameOffset[0].length; i++)	// loop over all offsets in this frame.
							{	
								//			System.out.println(frameOffset[0][i] + " x " + frameOffset[1][i]);
								if ((frameOffset[0][i]) < (meanX/counter - sigmaX*3) &&			// if we're not within 3 sigma away from mean.
										(frameOffset[0][i]) > (meanX/counter + sigmaX*3) &&
										(frameOffset[1][i]) < (meanY/counter - sigmaY*3) &&
										(frameOffset[1][i]) > (meanY/counter + sigmaY*3))
								{
									meanX -= frameOffset[0][i];		// remove this entry
									meanY -= frameOffset[1][i];		// remove this entry
									counter--;						// decrease.
								}
							}
						}
						if (counter > 0 && startIdx <inputParticle.size() && xyOffset.length > inputParticle.get(startIdx).frame - inputParticle.get(chStart).frame)
						{
							meanX /= counter;	// calculate mean.
							meanY /= counter;	// calculate mean
							xyOffset[0][inputParticle.get(startIdx).frame - inputParticle.get(chStart).frame][ch] = meanX; // remove this value from x for this z (or rather frame) and channel.
							xyOffset[1][inputParticle.get(startIdx).frame - inputParticle.get(chStart).frame][ch] = meanY; // remove this value from x for this z (or rather frame) and channel.
						}
					}
					//			System.out.println(idxCounter + " : "+ xyOffset[0][idxCounter][ch] + " x " + xyOffset[1][idxCounter][ch] + " : " + startIdx);
					startIdx++;	// step to the next frame.
					if (startIdx >= inputParticle.size())	// if we're done.
					{
						optimize = false;
					}
					else if ((inputParticle.get(startIdx).frame - inputParticle.get(chStart).frame) >= calibration.length-1)
						optimize = false;
					else if (inputParticle.get(startIdx).channel != ch + 1)	// if we've changed channel.
						optimize = false;					
				} // while(optimize)

			} // if there are any centers in the central slice.
		}
		
		/*
		 * interpolate
		 */
		for (int ch = 0; ch <  calibration[0].length; ch++)
		{
			for (int XY = 0; XY < 2; XY++)
			{
				for (int idx = 0; idx < calibration.length; idx++)
				{
					if (xyOffset[XY][idx][ch] == 0 && idx != nFrames)
					{
						if (idx == 0) // if the first entry is 0
						{
							int tempIdx = idx+1;
							int added = 0;
							double total = 0;
							while (added < 2 && tempIdx != calibration.length)
							{
								if (xyOffset[XY][tempIdx][ch] != 0)
								{
									added++;
									if (added == 1)
										total -= xyOffset[XY][tempIdx][ch];
									else
										total += xyOffset[XY][tempIdx][ch];
									//	System.out.println("adding " + xyOffset[XY][tempIdx][ch] + " from " + tempIdx);
								}
								tempIdx++;
							}

							tempIdx--;
							//System.out.println("corr: " + (tempIdx-idx)+ " total: " + total );
							total /= (tempIdx-idx);

							while (tempIdx >= idx)
							{
								if (tempIdx + 1< calibration.length && xyOffset[XY][tempIdx][ch] == 0)
								{
									xyOffset[XY][tempIdx][ch] = xyOffset[XY][tempIdx+1][ch] - total;	
									//System.out.println(xyOffset[XY][tempIdx][ch] + " from " + xyOffset[XY][tempIdx+1][ch] + " - " + total);

								}
								tempIdx--;
							}
						}else if (idx == calibration.length-1)
						{
							double total = xyOffset[XY][idx-1][ch] - xyOffset[XY][idx-2][ch];														
							xyOffset[XY][idx][ch] = xyOffset[XY][idx-1][ch] + total;
						}
						else // central part.
						{
							double total = -xyOffset[XY][idx-1][ch];
							int added = 1;
							int tempIdx = idx + 1;
							while (added < 2 && tempIdx != calibration.length)
							{
								if (xyOffset[XY][tempIdx][ch] != 0)
								{
									added++;
									total += xyOffset[XY][tempIdx][ch];								
								}
								tempIdx++;
							}
							if (tempIdx != calibration.length)
							{
								tempIdx--;
								total /= (tempIdx-idx);
								while (tempIdx >= idx)
								{
									if (xyOffset[XY][tempIdx][ch] == 0)
									{
										xyOffset[XY][tempIdx][ch] = xyOffset[XY][tempIdx+1][ch] - total;	
									}
									tempIdx--;
								}
							}else if (added == 1) // if we do not have a value at the end.
							{
								total = xyOffset[XY][idx-1][ch] - xyOffset[XY][idx-2][ch];
								tempIdx = idx + 1;
								while (tempIdx < calibration.length)
								{
									xyOffset[XY][tempIdx][ch] = xyOffset[XY][tempIdx-1][ch] + total;
									tempIdx++;
								}
							}
						}
					}
				}
			}
		}
		/*		for (int idx =0; idx < calibration.length; idx++)
		{
			System.out.println(idx + " : " + xyOffset[0][idx][0] + " x " + xyOffset[1][idx][0]);
		}
		 */	

		/*
		 * 5 point mean filtering.
		 */
		for (int ch = 0; ch < calibration[0].length; ch++)
		{
			double[] tempVector = new double[xyOffset[0].length];	// 5 point smoothed data.
			for (int XY = 0; XY < 2; XY++)
			{
				int idx = 0;										// index variable.				

				while (idx < tempVector.length)
				{
					// 5 point smoothing:
					if (idx == 0)
					{
						int included = 0;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 2][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx][ch]
									+ xyOffset[XY][idx + 1][ch] 
											+ xyOffset[XY][idx + 2][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;
						}
					}else if (idx == 1)
					{
						int included = 0;					
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 2][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 1][ch]
									+ xyOffset[XY][idx][ch]
											+ xyOffset[XY][idx + 1][ch] 
													+ xyOffset[XY][idx + 2][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}else if (idx == xyOffset[XY].length - 2)
					{
						int included = 0;
						if (xyOffset[XY][idx - 2][ch] != 0)
							included++;
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 2][ch]
									+ xyOffset[XY][idx - 1][ch]
											+ xyOffset[XY][idx][ch] 
													+ xyOffset[XY][idx + 1][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}else if (idx == xyOffset[XY].length - 1)
					{
						int included = 0;
						if (xyOffset[XY][idx - 2][ch] != 0)
							included++;
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 2][ch]
									+ xyOffset[XY][idx - 1][ch]
											+ xyOffset[XY][idx][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}else if (idx < xyOffset[XY].length - 2)
					{
						int included = 0;
						if (xyOffset[XY][idx - 2][ch] != 0)
							included++;
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 2][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 2][ch]
									+ xyOffset[XY][idx - 1][ch]
											+ xyOffset[XY][idx][ch] 
													+ xyOffset[XY][idx + 1][ch]
															+ xyOffset[XY][idx + 2][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}
					idx++;

				} // tempVector populated.
				for (int i = 0; i < tempVector.length; i++ ) // Transfer temp variable back to main vector.
					xyOffset[XY][i][ch] = tempVector[i];
			}// loop over x and y


		}// channel loop.

		/*
		Plot plot = new Plot("PRILM calibration", "z [nm]", "Angle [rad]");
		double[] zOffset = getZoffset();
		for (int ch = 0; ch < 2; ch++)
		{
			double[] printout = new double[calibration.length];
			double[] x = new double[printout.length];
			for (int i = 0; i < printout.length; i++)
			{
				printout[i] = xyOffset[ch][i][0];
				x[i] = (i-zOffset[0]);
			}
			if (ch == 0)
				plot.setColor(Color.BLACK);
			if (ch == 1)
				plot.setColor(Color.BLUE);
			if (ch == 2)
				plot.setColor(Color.RED);
			if (ch == 3)
				plot.setColor(Color.GREEN);
			if (ch == 4)
				plot.setColor(Color.MAGENTA);
			plot.addPoints(x,printout, Plot.LINE);
		}
		if (calibration[0].length == 1)
			plot.addLegend("Ch 1 \n Ch 2");
		if (calibration[0].length == 2)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3");		
		if (calibration[0].length == 3)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3 \n Ch 4");
		if (calibration[0].length == 4)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3 \n Ch 4 \n Ch 5");
		plot.show();
		/*	for (int i = 0; i < calibration.length; i++)
		{
			System.out.println("idx: " + i + " offset: " + xyOffset[0][i][0] + " x " +xyOffset[1][i][0]);
		}
		 */
		return xyOffset;
	}
	public static double[][][] getXYoffset2(ArrayList<Particle> inputParticle,double[][] calibration)
	{
		double[][][] xyOffset 	= new double[2][calibration.length][calibration[0].length]; // x-y (z,ch). 
		double[] center 		= getZoffset();			// get info of where z=0 is in the calibration file.

		for (int ch = 0; ch < calibration[0].length; ch++)	// over all channels.
		{
			int nCenter 	= 0; // number of locations to search after.
			for (int idx 	= 0; idx < inputParticle.size(); idx++)
			{				
				if (inputParticle.get(idx).frame == center[ch] && inputParticle.get(idx).channel == ch+1) // if we're at the frame corresponding to z = 0 and in the correct channel.
				{
					nCenter++;
				}
			}
			if (nCenter > 0)	// if we found centers.
			{
				double[][] localZeroXY = new double[2][nCenter];	// array to store all individual offsets.
				int counter  = 0;	
				for (int idx = 0; idx < inputParticle.size(); idx++)
				{
					if (inputParticle.get(idx).frame == center[ch] && inputParticle.get(idx).channel == ch+1)	// if we're at the center frame.
					{
						localZeroXY[0][counter] = inputParticle.get(idx).x;	// store x coordinate.
						localZeroXY[1][counter] = inputParticle.get(idx).y;	// store y coordinate.
						counter++;
					}
				}// localZeroXY now contains x-y coordinates for central slice particles.
				/*
				 * Find smallest offset against localZeroXY for each particle in stack for this channel. Take mean, remove outliers.
				 */
				int startIdx = 0; // which inded to start from.
				while (inputParticle.get(startIdx).channel != ch+1) // jump forward to first index for the current channel,
					startIdx ++;
				int endIdx		 = startIdx; // final index for this frame and channel.
				boolean optimize = true;	// loop whilst this is true.
				while (optimize)
				{
					counter = 0;	// keep track of number of added corrections.
					endIdx = startIdx;	// restart.
					boolean loop = true;	// exit flag for final frame and channel.
					if (inputParticle.size() > endIdx+1) // if we're not at the end of the list.
					{
						while (inputParticle.get(startIdx).frame == inputParticle.get(endIdx).frame && inputParticle.get(endIdx).channel == ch+1 && loop) // as long as we're within the same frame and channel.
						{
							if (inputParticle.size() > endIdx+1) // if we're not at the end
							{
								endIdx++;						 // step forward

							}else
								loop = false; // exit for final frame and channel.
						}
					}
					double[][] frameOffset = new double[2][endIdx-startIdx + 1]; // preallocate
					double meanX = 0;	// mean x offset.
					double meanY = 0;	// mean y offset.
					while (startIdx <= endIdx) // whilst we're not done with this frame.
					{
						double z = inputParticle.get(startIdx).z / ij.Prefs.get("SMLocalizer.calibration.DoubleHelix.step",0);  // get back approximate frame.
						z += center[ch];																						// shift by center to get frame number.
						if (z > (inputParticle.get(startIdx).frame - 3) && z < (inputParticle.get(startIdx).frame + 3)) 		// if we're within +/- 3 frames of the current one (ie if the fit worked with 3*zStep precision)
						{
							double distance = 200; // max distance to be considered in nm.
							for (int i = 0; i < localZeroXY[0].length; i ++) // loop over all particles from center frame.
							{
								double tempDistance = Math.sqrt((inputParticle.get(startIdx).x - localZeroXY[0][i])*(inputParticle.get(startIdx).x - localZeroXY[0][i]) + // calculate distance from current particle to the current center frame particle.
										(inputParticle.get(startIdx).y - localZeroXY[1][i])*(inputParticle.get(startIdx).y - localZeroXY[1][i]));
								if (tempDistance < distance) // if the particles are closer then the previous loops (or start value)
								{
									frameOffset[0][counter] = localZeroXY[0][i] - inputParticle.get(startIdx).x;	// update offset for this particle
									frameOffset[1][counter] = localZeroXY[1][i] - inputParticle.get(startIdx).y;	// update offset for this particle
									distance = tempDistance;														// update distance.
								}
							} // nearest neighbor found.
							meanX += frameOffset[0][counter]; 	// add offset in x to mean.
							meanY += frameOffset[1][counter]; 	// add offset in y to mean.
							counter ++;						 	// step forward in the array
						}
						startIdx++;								// next particle within the frame.
					}
					double sigmaX = 0;	// std of x offsets.
					double sigmaY = 0; 	// std of y offsets.
					if (counter > 0)	// if we've added values (ie found particles close enough to the center frame particles)
					{
						if (counter > 1) // sample requires more then 1 entry.
						{
							for (int i = 0; i < frameOffset[0].length; i++)	// loop over all offsets in this frame.
							{
								sigmaX += ((frameOffset[0][i] - meanX/counter)*(frameOffset[0][i] - meanX/counter))/(counter-1);	// add (Xi-µ)/(n-1)
								sigmaY += ((frameOffset[1][i] - meanY/counter)*(frameOffset[1][i] - meanY/counter))/(counter-1);	// add (Yi-µ)/(n-1)
							}
							sigmaX = Math.sqrt(sigmaX);	// square root of sigmax^2.
							sigmaY = Math.sqrt(sigmaY);	// square root of sigmay^2.
							for (int i = 0; i < frameOffset[0].length; i++)	// loop over all offsets in this frame.
							{								
								if ((frameOffset[0][i]) < (meanX/counter - sigmaX*3) &&			// if we're not within 3 sigma away from mean.
										(frameOffset[0][i]) > (meanX/counter + sigmaX*3) &&
										(frameOffset[1][i]) < (meanY/counter - sigmaY*3) &&
										(frameOffset[1][i]) > (meanY/counter + sigmaY*3))
								{
									meanX -= frameOffset[0][i];		// remove this entry
									meanY -= frameOffset[1][i];		// remove this entry
									counter--;						// decrease.
								}
							}
						}
						if (counter > 0)
						{
							meanX /= counter;	// calculate mean.
							meanY /= counter;	// calculate mean
							xyOffset[0][inputParticle.get(endIdx).frame-1][ch] = meanX; // remove this value from x for this z (or rather frame) and channel.
							xyOffset[1][inputParticle.get(endIdx).frame-1][ch] = meanY; // remove this value from x for this z (or rather frame) and channel.
						}
					}
					startIdx++;	// step to the next frame.
					if (startIdx >= inputParticle.size())	// if we're done.
						optimize = false;
					else if (inputParticle.get(startIdx).channel != ch + 1)	// if we've changed channel.
						optimize = false;					
				} // while(optimize)

			} // if there are any centers in the central slice.
		}
		/*
		 * 5 point mean filtering.
		 */
		for (int ch = 0; ch < calibration[0].length; ch++)
		{
			double[] tempVector = new double[xyOffset[0].length];	// 5 point smoothed data.
			for (int XY = 0; XY < 2; XY++)
			{
				int idx = 0;										// index variable.				

				while (idx < tempVector.length)
				{
					// 5 point smoothing:
					if (idx == 0)
					{
						int included = 0;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 2][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx][ch]
									+ xyOffset[XY][idx + 1][ch] 
											+ xyOffset[XY][idx + 2][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;
						}
					}else if (idx == 1)
					{
						int included = 0;					
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 2][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 1][ch]
									+ xyOffset[XY][idx][ch]
											+ xyOffset[XY][idx + 1][ch] 
													+ xyOffset[XY][idx + 2][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}else if (idx == xyOffset[XY].length - 2)
					{
						int included = 0;
						if (xyOffset[XY][idx - 2][ch] != 0)
							included++;
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 2][ch]
									+ xyOffset[XY][idx - 1][ch]
											+ xyOffset[XY][idx][ch] 
													+ xyOffset[XY][idx + 1][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}else if (idx == xyOffset[XY].length - 1)
					{
						int included = 0;
						if (xyOffset[XY][idx - 2][ch] != 0)
							included++;
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 2][ch]
									+ xyOffset[XY][idx - 1][ch]
											+ xyOffset[XY][idx][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}else if (idx < xyOffset[XY].length - 2)
					{
						int included = 0;
						if (xyOffset[XY][idx - 2][ch] != 0)
							included++;
						if (xyOffset[XY][idx - 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 1][ch] != 0)
							included++;
						if (xyOffset[XY][idx + 2][ch] != 0)
							included++;
						if (included > 0)
						{
							tempVector[idx] = (xyOffset[XY][idx - 2][ch]
									+ xyOffset[XY][idx - 1][ch]
											+ xyOffset[XY][idx][ch] 
													+ xyOffset[XY][idx + 1][ch]
															+ xyOffset[XY][idx + 2][ch])/included;
						}
						else
						{
							tempVector[idx] = 0;

						}
					}
					idx++;

				} // tempVector populated.
				for (int i = 0; i < tempVector.length; i++ ) // Transfer temp variable back to main vector.
					xyOffset[XY][i][ch] = tempVector[i];
			}// loop over x and y


		}// channel loop.


		Plot plot = new Plot("Double helix calibration", "z [nm]", "Angle [rad]");
		double[] zOffset = getZoffset();
		for (int ch = 0; ch < 2; ch++)
		{
			double[] printout = new double[calibration.length];
			double[] x = new double[printout.length];
			for (int i = 0; i < printout.length; i++)
			{
				printout[i] = xyOffset[ch][i][0];
				x[i] = (i-zOffset[0]);
			}
			if (ch == 0)
				plot.setColor(Color.BLACK);
			if (ch == 1)
				plot.setColor(Color.BLUE);
			if (ch == 2)
				plot.setColor(Color.RED);
			if (ch == 3)
				plot.setColor(Color.GREEN);
			if (ch == 4)
				plot.setColor(Color.MAGENTA);
			plot.addPoints(x,printout, Plot.LINE);
		}
		if (calibration[0].length == 1)
			plot.addLegend("Ch 1 \n Ch 2");
		if (calibration[0].length == 2)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3");		
		if (calibration[0].length == 3)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3 \n Ch 4");
		if (calibration[0].length == 4)
			plot.addLegend("Ch 1 \n Ch 2 \n Ch 3 \n Ch 4 \n Ch 5");
		plot.show();
		/*	for (int i = 0; i < calibration.length; i++)
		{
			System.out.println("idx: " + i + " offset: " + xyOffset[0][i][0] + " x " +xyOffset[1][i][0]);
		}
		 */
		return xyOffset;
	}
	/*
	 * Used in generation of calibration file. Smoothes out fitted results, 5 point moving window mean.
	 */
	public static double[][] makeCalibrationCurve(double[][] result, int minLength, int nChannels,boolean printout, boolean full,boolean store)
	{
		int[] start 	= new int[nChannels]; 	// start index for calibration, channel specific.
		int[] end 		= new int[nChannels]; 	// end index for calibration, channel specific.
		int maxCounter 	= 0;
		int channelIdx 	= 1;					// current channel.

		while (channelIdx <= nChannels)			// loop over all channels.
		{
			double[] tempVector = new double[result.length];	// 5 point smoothed data.

			int idx = 0;										// index variable.
			boolean iterate = true;								// loop variable.

			while (idx < result.length)
			{
				// 5 point smoothing:
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
					if (tempVector[idx] < tempVector[idx - 1]) 
						counter++;
					if (tempVector[idx] > tempVector[idx - 1])
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
				if (tempVector[0] > tempVector[1])
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
		if (store)
		{
			for (int Ch = 1; Ch <= nChannels; Ch++)
			{				
				ij.Prefs.set("SMLocalizer.calibration.DoubleHelix.center.Ch"+Ch,Math.round(result.length/2)-start[Ch-1]);			
			} 			
			ij.Prefs.savePreferences(); // store settings.
		}
		if (maxCounter < 0) // error check.
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

	/*
	 * Used in generation of calibration file. Smoothes out fitted results, 5 point moving window mean.
	 */
	public static double[][] interpolate(double[][] result, int minLength, int nChannels)
	{
		int[] start = new int[nChannels];
		int[] end 	= new int[nChannels];
		int counter = 0;
		int maxCounter = 0;
		int channelIdx = 1;
		while (channelIdx <= nChannels)
		{
			int idx = 0;
			boolean iterate = true;
			while (idx < result.length && iterate)
			{
				if (result[idx][channelIdx-1] < 0)
				{
					counter++;
					if (counter == minLength) // if we've passed the set number of points.
						start[channelIdx-1] = idx - minLength + 1;
					if (counter > minLength)
						end[channelIdx-1] = idx;
				}
				else if (result[idx][channelIdx-1] >= 0) 
				{
					if (counter < minLength)
						counter = 0;
					if (counter >= minLength)
						iterate = false;
				}
				idx++;
			}
			if (counter > maxCounter)
				maxCounter = counter;

			channelIdx++;
		}
		if (maxCounter >= minLength)
		{


			double[][] calibration = new double[maxCounter][nChannels];
			channelIdx = 1;

			while (channelIdx <= nChannels)
			{
				int idx = start[channelIdx-1];
				int count = 0;
				while (idx <= end[channelIdx-1])				
				{
					// 5 point smoothing:
					if (idx == start[channelIdx-1])
					{
						calibration[count][channelIdx-1] = (result[idx][channelIdx-1]
								+ result[idx + 1][channelIdx-1] 
										+ result[idx + 2][channelIdx-1])/3;
					}else if (idx == start[channelIdx-1] + 1)
					{
						calibration[count][channelIdx-1] = (result[idx - 1][channelIdx-1]
								+ result[idx][channelIdx-1]
										+ result[idx + 1][channelIdx-1] 
												+ result[idx + 2][channelIdx-1])/4;
					}else if (idx == end[channelIdx-1] - 1)
					{
						calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1] 
												+ result[idx + 1][channelIdx-1])/4;
					}else if (idx == end[channelIdx-1])
					{
						calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1])/3;
					}else if (idx < end[channelIdx-1] - 1)
					{
						calibration[count][channelIdx-1] = (result[idx - 2][channelIdx-1]
								+ result[idx - 1][channelIdx-1]
										+ result[idx][channelIdx-1] 
												+ result[idx + 1][channelIdx-1]
														+ result[idx + 2][channelIdx-1])/5;
					}
					count++;
					idx++;
				}
				channelIdx++;
			}
			return calibration;
		}

		else
		{
			double[][] calibration = new double[maxCounter][nChannels];
			return calibration;
		}
	}

}