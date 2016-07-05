import java.util.ArrayList;

public class cleanParticleList {

	public static void run(double[] lb, double[] ub, boolean[] Include){
		// Pull in user data.
		/*	Include[0] = inclXYsigma.isSelected();
        Include[1] = inclZsigma.isSelected();        
        Include[2] = inclXYprecision.isSelected();
        Include[3] = inclZprecision.isSelected();
        Include[4] = inclChiSquare.isSelected();
        Include[5] = inclPhotonCount.isSelected();

		 */

		ArrayList<Particle> Results = TableIO.Load();

		for (int i = 0; i < Results.size(); i++){
			int Remove = 0;
			if (Include[0]){
				if(Results.get(i).sigma_x >= lb[0] &&
						Results.get(i).sigma_x <= ub[0]){

				}else
					Remove++;
			}

			if (Include[0]){
				if(Results.get(i).sigma_y >= lb[1] &&			
						Results.get(i).sigma_y <= ub[1]){

				}else
					Remove++;
			}
			if (Include[1]){ 
				if(Results.get(i).sigma_z >= lb[2] &&
						Results.get(i).sigma_z <= ub[2]){

				}else
					Remove++;
			}
			if (Include[2]){
				if(Results.get(i).precision_x >= lb[3] &&
						Results.get(i).precision_x <= ub[3]){

				}else
					Remove++;
			}
			if (Include[2]){ 
				if(Results.get(i).precision_y >= lb[4] &&
						Results.get(i).precision_y <= ub[4]){

				}else
					Remove++;
			}
			if (Include[3]){ 
				if(Results.get(i).precision_z >= lb[5] &&
						Results.get(i).precision_z <= ub[5]){

				}else
					Remove++;
			}
			if (Include[4]){
				if(Results.get(i).chi_square >= lb[6] &&			
						Results.get(i).chi_square <= ub[6]){

				}else
					Remove++;
			}
			if (Include[5]){
				if(Results.get(i).photons >= lb[7] &&
						Results.get(i).photons <= ub[7]){

				}else
					Remove++;				
			}

			if(Remove == 0){
				Results.get(i).include = 1; // Include particle.
			}else {
				Results.get(i).include = 0; // Exclude particle
			}			
		}
		TableIO.Store(Results);

	}
	public static void delete(){ // Remove entries outside of parameter range, first call the run function.
		ArrayList<Particle> Results = TableIO.Load();
		ArrayList<Particle> cleanedList = new ArrayList<Particle>();
		for (int i = 0; i < Results.size(); i++){
			if (Results.get(i).include == 1){
				cleanedList.add(Results.get(i));
			}
		}
		TableIO.Store(cleanedList);
	}
}
