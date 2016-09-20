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
 * Particle is a object that hold all fitting information.
 */
public class Particle {
	double x, y, z, sigma_x, sigma_y, sigma_z, precision_x, precision_y, precision_z, r_square,photons;
	int include, frame, channel;
	public Particle(){
		this.x 				= 0; // Fitted x coordinate.
		this.y 				= 0; // Fitted y coordinate.
		this.z				= 0; // Fitted z coordinate.
		this.frame			= 0; // frame that the particle was identified.
		this.channel 		= 0; // Channel id.
		this.sigma_x 		= 0; // fitted sigma in x direction.
		this.sigma_y 		= 0; // fitted sigma in y direction.
		this.sigma_z 		= 0; // fitted sigma in z direction.
		this.precision_x 	= 0; // precision of fit for x coordinate.
		this.precision_y 	= 0; // precision of fit for y coordinate.
		this.precision_z 	= 0; // precision of fit for z coordinate.
		this.r_square 	= 0; // Goodness of fit.
		this.photons 		= 0; // Total photon count for this particle.
		this.include		= 0; // If this particle should be included in analysis and plotted.
	}
	
	public Particle(double x, double y, double z, int frame, int channel, double sigma_x, double sigma_y, double sigma_z, double precision_x, double precision_y, double precision_z, double r_square,double photons, int include){
		this.x 				= x; 			// Fitted x coordinate.
		this.y 				= y;			// Fitted y coordinate.
		this.z				= z; 			// Fitted z coordinate.
		this.frame			= frame; 		// frame that the particle was identified.
		this.channel 		= channel; 			// Channel id.
		this.sigma_x 		= sigma_x; 		// fitted sigma in x direction.
		this.sigma_y 		= sigma_y; 		// fitted sigma in y direction.
		this.sigma_z 		= sigma_z; 			// fitted sigma in z direction.
		this.precision_x 	= precision_x; 	// precision of fit for x coordinate.
		this.precision_y 	= precision_y; 	// precision of fit for y coordinate.
		this.precision_z 	= precision_z; 			// precision of fit for z coordinate.
		this.r_square 		= r_square; 	// Goodness of fit.
		this.photons 		= photons; 		// Total photon count for this particle.
		this.include		= include; 		// If this particle should be included in analysis and plotted.
	}
}