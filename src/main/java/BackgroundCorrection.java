import ij.ImageStack;
import ij.process.ImageProcessor;

/* This class contains all relevant algorithms for background corrections.
 * V 1.0 2016-06-18 Kristoffer Bernhem, kristoffer.bernhem@gmail.com
 */

class BackgroundCorrection {

	/* Main function for background filtering using the time median based method described in:
	 * The fidelity of stochastic single-molecule super-resolution reconstructions critically depends upon robust background estimation
	 *	E. Hoogendoorn, K. C. Crosby, D. Leyton-Puig, R. M.P. Breedijk, K. Jalink, T. W.J. Gadella & M. Postma
	 *	Scientific Reports 4, Article number: 3854 (2014)
	 *  Call through TestArray = BackgroundCorrection.medianFiltering(double TestArray[][][], int W); 
	 *  with W being filter window in one direction
	 *  and TestArray being a 3 dimensional array. Return is a in time running median filtered signal.
	 *  Alternatively with if another way of organizing the data is available, the median calculations can be accessed through:
	 *  vector = BackgroundCorrection.runningMedian(double[] Vector, int W); W is the filter window in one direction.
	 *  
	 *  
	 */

	public static ImageStack medianFiltering(ImageStack stack,int W){
		int nframes = stack.getSize();					// Number of timepoints.
		int rows = stack.getWidth(); 					// Width of each frame.
		int columns = stack.getHeight(); 				// Height of each frame.
		int pixelCount = rows*columns; 					// Pixels per frame.
		double[] MeanFrame = new double[nframes]; 		// Will include frame mean value.
		ImageProcessor IP = stack.getProcessor(1);		// get image processor for the stack.
		for (int Frame = 1; Frame < nframes+1; Frame++){			
			IP = stack.getProcessor(Frame); 			// Update processor to next slice.
			int[][] frameArray = IP.getIntArray();      // Get frame.
			for (int i = 0; i < rows; i++){ 			// Sum frame intensity.
				for (int j = 0; j < columns; j++){
					MeanFrame[Frame-1] += frameArray[i][j];
				}
			}			
			MeanFrame[Frame-1] /= pixelCount;			// Normalize frame intensity by pixelcount (get mean value).
		}
		
		ImageStack CorrectedStack = stack.duplicate(); 	// Make a copy of the input stack, user might want to keep the orignal.
		double[] voxels = new double[nframes];			// Hold values for each timepoint for a given xy set.
		double[] timeVector = new double[nframes]; 		// Hold values for each timepoint for a given xy set.
		// Calculate median
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < columns; j++){
				for (int k = 0; k < voxels.length; k++){
					voxels[k] = stack.getVoxel(i, j, k);
					timeVector[k] = voxels[k]/MeanFrame[k]; // Normalize voxels;
				}
				timeVector = runningMedian(timeVector, W); // Calculate time median for this xy position.
				for (int k = 0; k < voxels.length; k++){
					float data = ((float)(voxels[k] - timeVector[k]*MeanFrame[k]));
					if (data<0)
						data = 0;					
					CorrectedStack.setVoxel(i, j, k, data); // Update the duplicate stack with corrected values.
				}			
			}
		}
		return CorrectedStack;
	}

	public static double[] medianFiltering2(double[][][] IMstack,int W){

		int rows = IMstack.length; // Get input size in x.
		int cols = IMstack[0].length; // Get input size in y.
		int frames = IMstack[0][0].length; // Get input size in z.
		double[] MeanFrame = new double[frames]; // Will include frame mean value.
		for (int z = 0; z < frames; z ++){ // Loop over all frames, calculate frame median.
			double Sum = 0;
			for (int x = 0; x < rows; x ++){
				for (int y = 0; y < cols; y ++){
					Sum += IMstack[x][y][z] + 1e-20; // Add small number to avoid division by 0.
				}
			}
			MeanFrame[z] = Sum/(rows*cols);
		}

		double[] timeVector = new double[frames]; 
		double[] NoiseVector = new double[frames];
		for (int x = 0; x < rows; x ++){
			for (int y = 0; y < cols; y ++){

				for (int z = 0; z < frames; z ++){ // Loop over all frames, normalize IMstack.
					timeVector[z] = IMstack[x][y][z]/MeanFrame[z]; // Normalized values
				}			
				// This call could be parallelized over x and y on GPU for speedup, completely disconnected
				double[] timeVectorMedian = runningMedian(timeVector, W); // Calculate time median for this xy position.
							
				for (int z = 0; z < frames; z ++){ // Loop over all frames,
					IMstack[x][y][z] -= MeanFrame[z]*timeVectorMedian[z]; // Correct each pixel based on normalized time median.
					NoiseVector[z] += timeVectorMedian[z]; // Add up all pixelvalues
				}
				

			}	
		}
		int Pixels = rows*cols;
		for (int z = 0; z < frames; z++){			
			NoiseVector[z]  = MeanFrame[z]*NoiseVector[z]/Pixels;						
		}
		// Combine output into one array
		double[] OutputArray = new double[Pixels*frames+frames]; // All IMstack data followed by NoiseVector
		int x = 0;
		int y = 0;
		int z = 0;
		for (int i = 0; i < Pixels*frames;i++){
			OutputArray[i] = IMstack[x][y][z];
			x++;
			if (x == rows){
				x = 0; // Reset
				y++;				
			}
			if (y == cols){
				y = 0;
				z++;
			}
		}
		int Count = 0;
		for (int i = Pixels*frames; i < Pixels*frames + frames; i++){
			OutputArray[i] =  NoiseVector[Count];
			Count++;			
		}
		return OutputArray;
	}

	public static double[] runningMedian(double[] Vector, int W){
		// Preallocate variables.
		double[] medianVector = new double[Vector.length]; // Output vector.
		double[] V = new double[W+1];  // Vector for calculating running median.
//		double[] V2 = new double[2*W];  // Swap vector for removal of entries from V.
//		int idx = 0; 					// Preallocate, used for finding index of V which should be removed.		
		for(int i = 0; i <= W; i++){ // Transfer first 2xW+1 entries.
			V[i] = Vector[i];
		}

		int low = 0;
		int high = W-1;			   
		quickSort(V, low, high); // Quick version, sort first W entries.		

		for (int i = 0; i < W; i ++){ // First section, without access to data on left.//
			if (i % 2 == 0){
				medianVector[i] = (V[W/2+i/2]+V[W/2+i/2+1])/2.0;

			}else{
				medianVector[i] = V[W/2+i];
			}		
			V = sortInsert(V,Vector[i+W+1]); // Add new entry.

		}				
		
		// Something wrong, returns 0
		for(int i = W; i < Vector.length-W-1; i++){ // Main loop, middle section.			
			medianVector[i] = V[W]; // Pull out median value.			
			//V2 = removeEntry(V,TestVector[i-W]);  // Alternative to code below.
			V = removeEntry(V,Vector[i-W]);
			V = sortInsert(V,Vector[i+W+1]);
			/*
			idx =  indexOfIntArray(V, Vector[i-W]);
			if (idx == 0){
				System.arraycopy(V, 1, V2, 0, V2.length);
			}else if (idx == -1){
				System.arraycopy(V, 0, V2, 0, V2.length);
			}
			else if(idx == V2.length){ // Last element.
				System.arraycopy(V, 0, V2, 0, idx-1);
			}
			else{
				System.arraycopy(V, 0, V2, 0, idx-1);
				System.arraycopy(V, idx, V2, idx-1, V2.length-idx-1);				
			} 		
			// Code to replace removeEntry ends here.
			V = sortInsert(V2,Vector[i+W+1]);	*/
		}

		for (int i = Vector.length-W-1; i < Vector.length; i++){ // Last section, without access to data on right.			
			if (i % 2 == 0){				
				medianVector[i] = V[W-(i-Vector.length + W +1)/2];
			}else{
				medianVector[i] = (V[W-(i-Vector.length + W)/2]+V[W-(i-Vector.length + W)/2-1])/2.0;
			}
			V = removeEntry(V,Vector[i-W]); // Remove items from V once per loop, ending with a W+1 large vector.	
		}		
		return medianVector;
	}

	public static double[] removeEntry(double[] inVector, double entry) { // Return vector with element "entry" missing.
		int found = 0;
		double[] vectorOut = new double[inVector.length -1];
		for (int i = 0; i < inVector.length - 1;i++){
			if (inVector[i] == entry){
				found = 1;				
			}
			vectorOut[i] = inVector[i+found];
		}
		return vectorOut;
	} 

	public static int indexOfIntArray(double[] array, double key) {
		int returnvalue = -1;
		for (int i = 0; i < array.length; ++i) {
			if (key == array[i]) {
				returnvalue = i;
				break;
			}
		}
		return returnvalue;
	}

	public static double[] sortInsert(double[] Vector, double InsVal){ // Assumes sorted input vector.
		double[] bigVector = new double[Vector.length + 1]; // Add InsVal into this vector		
		int indexToInsert = 0;
		if (InsVal > Vector[Vector.length-1]){ // If the value to be inserted is larger then the last value in the vector.
			/*			for (int i = 0; i < Vector.length; i++){
				bigVector[i] = Vector[i];
			} */									
			System.arraycopy(Vector, 0, bigVector, 0, Vector.length);
			bigVector[bigVector.length-1] = InsVal;
			return bigVector;
		}else if (InsVal < Vector[0]){  // If the value to be inserted is smaller then the first value in the vector.
			bigVector[0] = InsVal;
			System.arraycopy(Vector, 0, bigVector, 1, Vector.length);
			/*			for (int i = 1; i <= Vector.length; i++){
				bigVector[i] = Vector[i-1];
			}*/
			return bigVector;
		}else{
			for (int i = 1; i < Vector.length; i++){
				if (InsVal < Vector[i]){
					indexToInsert = i;

					System.arraycopy(Vector, 0, bigVector, 0, indexToInsert);
					bigVector[indexToInsert] = InsVal;
					System.arraycopy(Vector, indexToInsert, bigVector, indexToInsert+1, Vector.length-indexToInsert);
					return bigVector;
				}

			}
		}
		return bigVector;
	}

	public static void quickSort(double[] arr, int low, int high) {
		if (arr == null || arr.length == 0)
			return;

		if (low >= high)
			return;

		// pick the pivot
		int middle = low + (high - low) / 2;
		double pivot = arr[middle];

		// make left < pivot and right > pivot
		int i = low, j = high;
		while (i <= j) {
			while (arr[i] < pivot) {
				i++;
			}

			while (arr[j] > pivot) {
				j--;
			}

			if (i <= j) {
				double temp = arr[i];
				arr[i] = arr[j];
				arr[j] = temp;
				i++;
				j--;
			}
		}

		// recursively sort two sub parts
		if (low < j)
			quickSort(arr, low, j);

		if (high > i)
			quickSort(arr, i, high);
	}
}