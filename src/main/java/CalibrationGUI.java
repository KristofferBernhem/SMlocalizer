

import javax.swing.JOptionPane;

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
		algorithmSelect.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "PRILM", "Double Helix", "Biplane" , "Astigmatism", "2D"}));

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

		int zStep = Integer.parseInt(JOptionPane.showInputDialog("z step for calibration file? [nm]",
				"10"));

		if(algorithmSelect.getSelectedIndex() == 0)
		{ // PRILM			
			int inputPixelSize 	= Integer.parseInt(pixelSize.getText());		
			PRILMfitting.calibrate(inputPixelSize, zStep);

		}else if(algorithmSelect.getSelectedIndex() == 1)
		{ // DH			
			DoubleHelixFitting.calibrate(Integer.parseInt(pixelSize.getText()), zStep);
		}else 	
			if(algorithmSelect.getSelectedIndex() == 2)
			{ // Biplane
				BiplaneFitting.calibrate(Integer.parseInt(pixelSize.getText()), zStep);			
			}else
				if(algorithmSelect.getSelectedIndex() == 3)
				{ // Astigmatism
					AstigmatismFitting.calibrate(Integer.parseInt(pixelSize.getText()),zStep);
				}
				else if(algorithmSelect.getSelectedIndex() == 4)
				{
					int[] gain = {100};
					BasicFittingCorrections.calibrate(100,gain);
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
