package ca.germuth.machine_learning.univariate_linear_regression;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {

	private static final double LEARNING_RATE = 0.33;
	private static final String TRAINING_FILE = "training_data.txt";
	
	public static void main(String[] args) {
		DecimalFormat df = new DecimalFormat();
		df.setMaximumFractionDigits(3);
		
		//hypothesis function is y = Mx + B
		//need to find best M and B
		//start with random?
		double M = 2;
		double B = 2;
		
		
		ArrayList<TrainingData> training = readTrainingData();

		double currError = error(M, B, training);
		
		//difficult to test for convergence, so rather
		//just do 100 fixed iterations
		for(int i = 0; i < 100; i++){
			//take derivative of cost function
			//see https://chrisjmccormick.wordpress.com/2014/03/04/gradient-descent-derivation/
			
			// dJ/dB = 1/n * sum( predicted - actual )
			double derivativeB = 0.0;
			for(TrainingData td: training){
				//				predicted - actual
				derivativeB += (M*td.input + B) - td.output;
			}
			derivativeB /= training.size();
			
			// dJ/dM = 1/n * sum( predicted - actual ) * x(i)
			double derivativeM = 0.0;
			for(TrainingData td: training){
				//				(predicted - actual)		  *  input
				derivativeM += ((M*td.input + B) - td.output) * td.input;
			}
			derivativeM /= training.size();
			
			M = M - LEARNING_RATE * derivativeM;
			B = B - LEARNING_RATE * derivativeB;
			
			currError = error(M, B, training);			
		}
		
		//display results
		System.out.println("Best Estimate for Equation:");
		System.out.println("y = " + df.format(M) + "x + " + df.format(B));
		System.out.println("");
		System.out.println("Output across Training Data:");
		for(TrainingData train: training){
			System.out.println("input: " + train.input + " output: " + train.output);
			System.out.println("y = " + df.format(M) + " * " + train.input + " + " + df.format(B) + " = " + df.format(M*train.input + B));
			System.out.println("Difference: " + df.format(train.output - (M*train.input + B)));
		}
		System.out.println("");
		System.out.println("Total Error:");
		System.out.println(df.format(error(M, B, training)));
	}

	private static double error(double M, double B, ArrayList<TrainingData> training){
		double error = 0.0;
		
		for(TrainingData train: training){
			double predicted = M*train.input + B;
			double actual = train.output;
			double diff = predicted - actual;
			error += Math.pow(diff, 2);
		}
		return error / (2 * training.size());
	}
	
	public static ArrayList<TrainingData> readTrainingData(){
		ArrayList<TrainingData> list = new ArrayList<TrainingData>();
		try {
			Scanner s = new Scanner(new File(TRAINING_FILE));
			
			//skip over header
			s.nextLine();
			s.nextLine();
			
			while(s.hasNextLine()){
				String line = s.nextLine();
				Scanner lineScanner = new Scanner(line);
				list.add(new TrainingData(lineScanner.nextInt(), lineScanner.nextInt()));
			}
			s.close();
		} catch (FileNotFoundException e) {
			System.err.println("File not Found");
			e.printStackTrace();
		}
		return list;
	}
	
}class TrainingData{
	public int input;
	public int output;
	public TrainingData(int i, int o){
		input = i;
		output = o;
	}
}
