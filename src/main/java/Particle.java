/*
 * Particle is a object that hold all fitting information.
 */
public class Particle {
	double x, y, frame, sigma_x, sigma_y, precision_x, precision_y, chi_square,photons;

	public Particle(){
		this.x 				= 0; // Fitted x coordinate.
		this.y 				= 0; // Fitted y coordinate.
		this.frame			= 0; // frame that the particle was identified.
		this.sigma_x 		= 0; // fitted sigma in x direction.
		this.sigma_y 		= 0; // fitted sigma in y direction.
		this.precision_x 	= 0; // precision of fit for x coordinate.
		this.precision_y 	= 0; // precision of fit for y coordinate.
		this.chi_square 	= 0; // Goodness of fit.
		this.photons 		= 0; // Total photon count for this particle.
	}
	
	public Particle(double x, double y, double frame, double sigma_x, double sigma_y, double precision_x, double precision_y, double chi_square,double photons){
		this.x 				= x; // Fitted x coordinate.
		this.y 				= y; // Fitted y coordinate.
		this.frame			= frame; // frame that the particle was identified.
		this.sigma_x 		= sigma_x; // fitted sigma in x direction.
		this.sigma_y 		= sigma_y; // fitted sigma in y direction.
		this.precision_x 	= precision_x; // precision of fit for x coordinate.
		this.precision_y 	= precision_y; // precision of fit for y coordinate.
		this.chi_square 	= chi_square; // Goodness of fit.
		this.photons 		= photons; // Total photon count for this particle.
	}
}