import java.util.ArrayList;

import ij.ImagePlus;
import ij.WindowManager;
import ij.process.ImageProcessor;
import ij.process.ImageStatistics;

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
 * @author kristoffer.bernhem
 */
@SuppressWarnings("serial")
public class CalibrationGUI extends javax.swing.JFrame {

	/**
	 * Creates new form NewJFrame
	 */
	public CalibrationGUI() {
		initComponents();
	}


	// <editor-fold defaultstate="collapsed" desc="Generated Code">                          
	private void initComponents() {

		jLabel1 = new javax.swing.JLabel();
		algorithmSelect = new javax.swing.JComboBox<>();
		jLabel2 = new javax.swing.JLabel();
		jLabel3 = new javax.swing.JLabel();
		pixelSize = new javax.swing.JTextField();
		stepSize = new javax.swing.JTextField();
		Process = new javax.swing.JButton();

		//  setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

		jLabel1.setFont(new java.awt.Font("Times New Roman", 0, 24)); // NOI18N
		jLabel1.setText("SMLocalizer 3D calibration");

		algorithmSelect.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
		algorithmSelect.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "PRILM", "Double Helix", "Biplane" , "Astigmatism"}));

		jLabel2.setFont(new java.awt.Font("Times New Roman", 0, 11)); // NOI18N
		jLabel2.setText("Pixelsize [nm]:");

		jLabel3.setFont(new java.awt.Font("Times New Roman", 0, 11)); // NOI18N
		jLabel3.setText("Step size Z [nm]:");

		pixelSize.setFont(new java.awt.Font("Times New Roman", 0, 11)); // NOI18N
		pixelSize.setHorizontalAlignment(javax.swing.JTextField.CENTER);
		pixelSize.setText("100");
		pixelSize.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				pixelSizeActionPerformed(evt);
			}
		});

		stepSize.setFont(new java.awt.Font("Times New Roman", 0, 11)); // NOI18N
		stepSize.setHorizontalAlignment(javax.swing.JTextField.CENTER);
		stepSize.setText("10");

		Process.setFont(new java.awt.Font("Times New Roman", 3, 12)); // NOI18N
		Process.setText("Process");
		Process.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				ProcessActionPerformed(evt);
			}
		});

		javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
		getContentPane().setLayout(layout);
		layout.setHorizontalGroup(
				layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addGroup(layout.createSequentialGroup()
						.addContainerGap()
						.addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
								.addComponent(algorithmSelect, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
								.addGroup(layout.createSequentialGroup()
										.addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
												.addComponent(jLabel3)
												.addComponent(jLabel2))
										.addGap(18, 18, 18)
										.addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
												.addComponent(pixelSize, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
												.addComponent(stepSize, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)))
								.addComponent(Process)
								.addComponent(jLabel1))
						.addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
				);
		layout.setVerticalGroup(
				layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addGroup(layout.createSequentialGroup()
						.addContainerGap()
						.addComponent(jLabel1)
						.addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
						.addComponent(algorithmSelect, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
						.addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
						.addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
								.addComponent(jLabel2)
								.addComponent(pixelSize, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
						.addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
						.addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
								.addComponent(jLabel3)
								.addComponent(stepSize, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
						.addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
						.addComponent(Process)
						.addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
				);

		pack();
	}// </editor-fold>                        

	private void pixelSizeActionPerformed(java.awt.event.ActionEvent evt) {                                          
	}                                         

	private void ProcessActionPerformed(java.awt.event.ActionEvent evt) {                               
		/*
		 * Localize all events in all frames.
		 */

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

		/*
		 * Set z values based on frame and z-stepsize.
		 */
		/*
		 * translate into calibration data based on method:
		 */


		if(algorithmSelect.getSelectedIndex() == 0)
		{ // PRILM			

			int inputPixelSize 	= Integer.parseInt(pixelSize.getText());
			int zStep = Integer.parseInt(stepSize.getText());
			PRILMfitting.calibrate(inputPixelSize, zStep);
		
		}else if(algorithmSelect.getSelectedIndex() == 1)
		{ // DH
			ImageStatistics IMstat 	= image.getStatistics();
			int[] MinLevel 			= {(int) (IMstat.max*0.3)};	
			double maxSigma 		= 2.5;
			System.out.println(MinLevel[0]);		
			int[] gWindow 			= {5};
			int[] inputPixelSize 	= {Integer.parseInt(pixelSize.getText())};
			int[] minPosPixels 		= {20};
			int[] totalGain 		= {100};
			int selectedModel 		= 0; // CPU
			System.out.println("Double helix"); 
			localizeAndFit.run(MinLevel, gWindow, inputPixelSize, minPosPixels, totalGain, selectedModel,maxSigma);
			/*
			 * clean out fits based on goodness of fit:
			 */
			boolean[][] include = new boolean[7][1];
			include[0][0] 		= false;
			include[1][0] 		= false;
			include[2][0] 		= false;
			include[3][0] 		= true;
			include[4][0] 		= false;
			include[5][0] 		= false;
			include[6][0] 		= false;    	
			double[][] lb 		= new double[7][1];
			lb[0][0]			= 0;
			lb[1][0]			= 0;
			lb[2][0]			= 0;
			lb[3][0]			= 0.8;
			lb[4][0]			= 0;
			lb[5][0]			= 0;
			lb[6][0]			= 0;
			double[][] ub 		= new double[7][1];
			ub[0][0]			= 0;
			ub[1][0]			= 0;
			ub[2][0]			= 0;
			ub[3][0]			= 1.0;
			ub[4][0]			= 0;
			ub[5][0]			= 0;
			ub[6][0]			= 0;
			cleanParticleList.run(lb,ub,include);
			cleanParticleList.delete();
			ArrayList<Particle> result = TableIO.Load();
			int zStep = Integer.parseInt(stepSize.getText());
			for (int i = 0; i < result.size(); i++)
			{
				result.get(i).z = (result.get(i).frame-1)*zStep;
			}
			TableIO.Store(result);
			result = TableIO.Load();
			int maxSqdist = 650*650;
			System.out.println("PRILM");
			System.out.println("Double helix");   
			int id = 2;		
			double[] angle = new double[nFrames];
			double[] distance = new double[nFrames];
			int[] count = new int[nFrames];
			for (int i = 0; i < result.size(); i++)
			{
				if (result.get(i).include == 1) // if the current entry is within ok range and has not yet been assigned.
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
								angle[(int)(result.get(i).frame-1)] += (Math.atan2(dy, dx)); // angle between points and horizontal axis.
								if (Math.sqrt(dx*dx + dy*dy) > distance[(int)(result.get(i).frame-1)])
									distance[(int)(result.get(i).frame-1)] = Math.sqrt(dx*dx + dy*dy);
								count[(int)(result.get(i).frame-1)]++;
								if (result.get(i).frame == 138)
								{
									System.out.println((Math.atan2(dy, dx)));
								}
							}
						}
						idx ++;
					}    				    				
					id++;
				}
			}
			for(int i = 0; i < count.length; i++)
			{
				if (count[i]>0)
				{
					angle[i] 	/= count[i]; // mean angle for this z-depth.					
				}
			}
			// interpolate to smooth out calibration curve.
			// Store calibrationfile.

			correctDrift.plot(angle);
			correctDrift.plot(distance);
			int start 	= 0;
			int end 	= 0;
			for (int i = 0; i < angle.length; i++) // loop over all and determine start and end
			{

			}


		}else 	
			if(algorithmSelect.getSelectedIndex() == 2)
			{ // Biplane
				/*
				 * Fit all and then calculate intensity distribution between the two cameras for z position. Use strongest intensity fit for x-y.
				 */			
				ImageStatistics IMstat 	= image.getStatistics();
				int[] MinLevel 			= {(int) (IMstat.max*0.3)};	
				double maxSigma = 6;
				System.out.println(MinLevel[0]);		
				int[] gWindow 			= {5}; // Astigmatism: 15. PRILM: 5
				int[] inputPixelSize 	= {Integer.parseInt(pixelSize.getText())};
				int[] minPosPixels 		= {20};
				int[] totalGain 		= {100};
				int selectedModel 		= 0; // CPU
				localizeAndFit.run(MinLevel, gWindow, inputPixelSize, minPosPixels, totalGain, selectedModel,maxSigma);
				/*
				 * clean out fits based on goodness of fit:
				 */
				boolean[][] include = new boolean[7][1];
				include[0][0] 		= false;
				include[1][0] 		= false;
				include[2][0] 		= false;
				include[3][0] 		= true;
				include[4][0] 		= false;
				include[5][0] 		= false;
				include[6][0] 		= false;    	
				double[][] lb 		= new double[7][1];
				lb[0][0]			= 0;
				lb[1][0]			= 0;
				lb[2][0]			= 0;
				lb[3][0]			= 0.8;
				lb[4][0]			= 0;
				lb[5][0]			= 0;
				lb[6][0]			= 0;
				double[][] ub 		= new double[7][1];
				ub[0][0]			= 0;
				ub[1][0]			= 0;
				ub[2][0]			= 0;
				ub[3][0]			= 1.0;
				ub[4][0]			= 0;
				ub[5][0]			= 0;
				ub[6][0]			= 0;
				cleanParticleList.run(lb,ub,include);
				System.out.println("Biplane");			
				ArrayList<Particle> result = TableIO.Load();
				double[] ratio = new double[nFrames];
				int[] count = new int[nFrames];
				int zStep = Integer.parseInt(stepSize.getText());
				for (int i = 0; i < result.size(); i++)
				{
					result.get(i).z = (result.get(i).frame-1)*zStep;
				}
				TableIO.Store(result);
				result = TableIO.Load();
				int maxSqdist = 500*500;
				double minRsquare = 0.8;
				int shift = image.getWidth()/2; // half the width is for the first camera, the second half is for the scond camera.
				shift *= inputPixelSize[0]; // translate to nm.

				for (int i = 0; i < result.size(); i++)
				{
					if (result.get(i).r_square >= minRsquare)
					{

						if (result.get(i).x < shift)
						{

							for (int j = 0; j < result.size(); j++) // loop over all entries
								if(result.get(j).include == 1 && result.get(j).x > shift && result.get(j).frame == result.get(i).frame)
								{								
									double dist = (result.get(i).x - result.get(j).x+shift)*(result.get(i).x - result.get(j).x+shift) + 
											(result.get(i).y - result.get(j).y)*(result.get(i).y - result.get(j).y);
									if (dist < maxSqdist)
									{		
										double base	= ( result.get(j).photons + result.get(i).photons);
										ratio[(int)(result.get(i).z/zStep)] += result.get(i).photons/base;
										//									ratio[(int)(result.get(i).z/zStep)] += result.get(i).photons/result.get(j).photons;
										count[(int)(result.get(i).z/zStep)]++;
									}

								}
						}
						else 
						{

							for (int j = 0; j < result.size(); j++) // loop over all entries
								if(result.get(j).include == 1 && result.get(j).x < shift && result.get(j).frame == result.get(i).frame)
								{								
									double dist = (result.get(i).x - result.get(j).x + shift)*(result.get(i).x - result.get(j).x + shift) + 
											(result.get(i).y - result.get(j).y)*(result.get(i).y - result.get(j).y);

									if (dist < maxSqdist)
									{
										double base	= ( result.get(j).photons + result.get(i).photons);							
										ratio[(int)(result.get(i).z/zStep)] += result.get(j).photons/base;
										//									ratio[(int)(result.get(i).z/zStep)] += result.get(j).photons/result.get(i).photons;
										count[(int)(result.get(i).z/zStep)]++;
									}

								}
						}

					}
				}
				for (int i = 0; i < nFrames; i++)
				{
					if (count[i]>0)
						ratio[i] /= count[i]; // normalize.
				}
				correctDrift.plot(ratio);
			}else
				if(algorithmSelect.getSelectedIndex() == 3)
				{ // Astigmatism
					/*
					 * requires rewrite of fit algorithm to handle larger values of sigma. 
					 */
					ImageStatistics IMstat 	= image.getStatistics();
					int[] MinLevel 			= {(int) (IMstat.max*0.1)};	
					double maxSigma 		= 7;
					System.out.println(MinLevel[0]);		
					int[] gWindow 			= {15}; 
					int[] inputPixelSize 	= {Integer.parseInt(pixelSize.getText())};
					int[] minPosPixels 		= {25};
					int[] totalGain 		= {100};
					int selectedModel 		= 0; // CPU
					localizeAndFit.run(MinLevel, gWindow, inputPixelSize, minPosPixels, totalGain, selectedModel,maxSigma);
					/*
					 * clean out fits based on goodness of fit:
					 */
					boolean[][] include = new boolean[7][1];
					include[0][0] 		= false;
					include[1][0] 		= false;
					include[2][0] 		= false;
					include[3][0] 		= true;
					include[4][0] 		= false;
					include[5][0] 		= false;
					include[6][0] 		= false;    	
					double[][] lb 		= new double[7][1];
					lb[0][0]			= 0;
					lb[1][0]			= 0;
					lb[2][0]			= 0;
					lb[3][0]			= 0.9;
					lb[4][0]			= 0;
					lb[5][0]			= 0;
					lb[6][0]			= 0;
					double[][] ub 		= new double[7][1];
					ub[0][0]			= 0;
					ub[1][0]			= 0;
					ub[2][0]			= 0;
					ub[3][0]			= 1.0;
					ub[4][0]			= 0;
					ub[5][0]			= 0;
					ub[6][0]			= 0;
					cleanParticleList.run(lb,ub,include);
					cleanParticleList.delete();
					ArrayList<Particle> result = TableIO.Load();
					int zStep = Integer.parseInt(stepSize.getText());
					for (int i = 0; i < result.size(); i++)
					{
						result.get(i).z = (result.get(i).frame-1)*zStep;
					}
					TableIO.Store(result);
					result = TableIO.Load();
					System.out.println("Astigmatism");
					double[] ratio = new double[nFrames];
					int[] count = new int[nFrames];
					for (int i = 0; i < result.size(); i++)
					{
						ratio[(int)(result.get(i).z/zStep)] += result.get(i).sigma_x/result.get(i).sigma_y;
						count[(int)(result.get(i).z/zStep)]++;
					}
					for (int i = 0; i < nFrames; i++)
					{
						if (count[i]>0)
							ratio[i] /= count[i]; // normalize.
					}
					correctDrift.plot(ratio);
				}
	}                                       

	public static double[] interpolate(double[] result, int minLength)
	{
		int start 	= 0;
		int end 	= 0;
		int counter = 0;
		for (int i = 0; i < result.length; i++) // loop over all and determine start and end
		{
			if (result[i] < 0)
			{
				counter++;
			}
			if (result[i] >= 0)
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
		double[] calibration = new double[end-start+1];

		for (int i = 0; i < calibration.length; i++)
		{
			if (i == 0) 
				calibration[i] = (result[start] + result[start + 1] + result[start + 2]) / 3;
			else if (i == 1)
				calibration[i] = (result[start] + result[start + 1] + result[start + 2] + result[start + 3]) / 4;
			else if (i == calibration.length-2)
				calibration[i] = (result[start + i + 1] + result[start + i] + result[start + i - 1] + result[start + i - 2])/4;
			else if (i == calibration.length-1)
				calibration[i] = (result[start + i] + result[start + i - 1]+ result[start + i - 2])/3;
			else
				calibration[i] = (result[start + i] + result[start + i - 1] + result[start + i + 1] + result[start + i - 2] + result[start + i + 2])/5;
		}
		return calibration;
	}
	/**
	 * @param args the command line arguments
	 */
	public static void main(String args[]) {
		/* Set the Nimbus look and feel */
		//<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
		/* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
		 * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
		 */
		try {
			for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
				if ("Nimbus".equals(info.getName())) {
					javax.swing.UIManager.setLookAndFeel(info.getClassName());
					break;
				}
			}
		} catch (ClassNotFoundException ex) {
			java.util.logging.Logger.getLogger(CalibrationGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		} catch (InstantiationException ex) {
			java.util.logging.Logger.getLogger(CalibrationGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		} catch (IllegalAccessException ex) {
			java.util.logging.Logger.getLogger(CalibrationGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		} catch (javax.swing.UnsupportedLookAndFeelException ex) {
			java.util.logging.Logger.getLogger(CalibrationGUI.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
		}
		//</editor-fold>
		//</editor-fold>
		//</editor-fold>
		//</editor-fold>

		/* Create and display the form */
		java.awt.EventQueue.invokeLater(new Runnable() {
			public void run() {
				new CalibrationGUI().setVisible(true);
			}
		});
	}

	// Variables declaration - do not modify                     
	private javax.swing.JButton Process;
	private javax.swing.JComboBox<String> algorithmSelect;
	private javax.swing.JLabel jLabel1;
	private javax.swing.JLabel jLabel2;
	private javax.swing.JLabel jLabel3;
	private javax.swing.JTextField pixelSize;
	private javax.swing.JTextField stepSize;
	// End of variables declaration                   
}
