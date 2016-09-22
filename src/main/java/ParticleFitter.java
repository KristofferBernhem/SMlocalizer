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
 * ParticleFitter.Fitter returns an arraylist of particles that has been fitted. Input is a ImageProcessor for a frame of interest, an int array list of center coordinates of interest,
 * Window width of square centered on these coordinates in pixels and frame number.
 */

public class ParticleFitter {

	
	public static Particle Fitter(fitParameters fitThese){ // setup a single gaussian fit, return localized particle.
		double convergence	= 1E-8;	// stop optimizing once improvement is below this.
		int maxIteration 	= 1000;	// max number of iterations.
		GaussSolver Gsolver = new GaussSolver(
				fitThese.data, 		// data to be fitted.
				fitThese.windowWidth, // window used for data extraction.
				convergence,
				maxIteration, 
				fitThese.Center, 		// center coordianates for center pixel.
				fitThese.channel, 		// channel id.
				fitThese.pixelsize,		// pixelsize in nm.
				fitThese.frame,			// frame number.
				fitThese.totalGain);	// total gain, camera specific parameter giving relation between input photon to output pixel intensity.
		Particle Results 	= Gsolver.Fit();	// do fit.
		return Results;
	}
}
