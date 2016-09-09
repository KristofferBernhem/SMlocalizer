
public class TimingTest {

	public static void main(String[] args) {
		
		int[] data = {
				3296, 4544,  5600,  5536,  5248,  4448, 3328,
				3760, 5344,  8240,  9680, 10592,  7056, 3328,
				3744, 6672, 14256, 24224, 20256, 11136, 5248,
				3696, 7504, 16944, 26640, 21680, 10384, 5008,
				2992, 6816, 10672, 15536, 14464,  7792, 4016,
				2912, 3872,  4992,  6560,  6448,  4896, 3392,
				3088, 3248,  3552, 	3504,  4144,  4512, 2944  
		};
		
		int[] data2 = { // slize 45 SingleBead2
				3888, 3984,  6192,   4192, 3664,  3472, 3136,
				6384, 8192,  12368, 12720, 6032,  5360, 3408, 
				6192, 13760, 21536, 20528, 9744,  6192, 2896,
				6416, 15968, 25600, 28080, 12288, 4496, 2400,
				4816, 11312, 15376, 14816, 8016,  4512, 3360,
				2944, 4688,  7168,   5648, 5824,  3456, 2912,
				2784, 3168,  4512,   4192, 3472,  2768, 2912
		};
		
		int[] Center = {5,5};
		int frame = 1;
		int pixelsize = 100;
		int windowWidth = 7;
		int channel = 1;
	int n = 1000;
	long start = System.nanoTime();
	for (int i = 0; i < n; i ++){
		fitParameters fitThese = new fitParameters(Center,data, channel, frame, pixelsize,  windowWidth);
		ParticleFitter.Fitter(fitThese);
	}
	long stop = System.nanoTime() - start;
	long startLM = System.nanoTime();
	for (int i = 0; i < n; i ++){
		fitParameters fitThese = new fitParameters(Center,data, channel, frame, pixelsize,  windowWidth);
		ParticleFitter.FitterLM(fitThese);
	}
	long stopLM = System.nanoTime() - startLM;

	System.out.println("Adaptive " + stop/1000000 + " LM: " + stopLM/1000000 );
	
	fitParameters fitThese = new fitParameters(Center,data, channel, frame, pixelsize,  windowWidth);
	Particle P = ParticleFitter.Fitter(fitThese);
	System.out.println("Adaptive: " + P.x + " x " + P.y);
	P= ParticleFitter.FitterLM(fitThese);
	System.out.println("LM: " + P.x + " x " + P.y);
	
	}
}
