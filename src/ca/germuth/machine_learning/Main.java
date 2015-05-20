package ca.germuth.machine_learning;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Multi Variate Linear Regression
 * 	Currently has Potential to use Gradient Descent or the Normal Equation
 * 	to solve multi variable linear regression
 * @author Aaron
 *
 */
public class Main {
	private static final String TRAINING_FILE = "training_data_3.txt";
	
	public static void main(String[] args) {
		DecimalFormat df = new DecimalFormat();
		df.setMaximumFractionDigits(3);
		
		ArrayList<TrainingData> training = readTrainingData();
//		training = normalize(training);
		training = addFakeFeature(training);

		double[] theta = GradientDescent.run(training, true);
//		double[] theta = NormalEquation.run(training);
		
		//display results
		System.out.println("Best Estimate for Equation:");
		String str = "y = ";
		for(int i = 0; i < theta.length; i++){
			str += df.format(theta[i]) + " ";
			if(i != 0){
				str += "x" + (i) + " ";
			}
			str += "+ ";
		}
		
		int numFeatures = training.get(0).input.length;
		System.out.println(str.substring(0, str.length() - 3));
		System.out.println("");
		System.out.println("Total Error:");
		System.out.println(df.format(error(theta, training)));
		System.out.println("");
		System.out.println("Press Y to predict an input, and N to close program");
		Scanner s = new Scanner(System.in);
		String token = s.next().toLowerCase();
		while(!token.contains("n")){
			System.out.println("Enter Input Features");
			double[] in = new double[numFeatures];
			in[0] = 1.0;
			for(int i = 1; i < numFeatures; i++){
				in[i] = s.nextDouble();
			}
			//TODO need to normalize input features if training set is normalized
			System.out.println("Expected Output is " + predict(theta, in));
		}
		s.close();
	}

	public static double predict(double[] theta, double[] input){
		//add extra one to input
		double result = 0.0;
		for(int i = 0; i < theta.length; i++){
			result += theta[i] * input[i];
		}
		return result;
	}
	
	public static double error(double[] theta, ArrayList<TrainingData> training){
		double error = 0.0;
		
		for(TrainingData train: training){
			double predicted = predict(theta, train.input);
			double actual = train.output;
			double diff = predicted - actual;
			error += Math.pow(diff, 2);
		}
		return error / (2 * training.size());
	}
	
	private static ArrayList<TrainingData> addFakeFeature(ArrayList<TrainingData> training) {
		int numFeatures = training.get(0).input.length;
		// now add ones for fake first feature
		ArrayList<TrainingData> result = new ArrayList<TrainingData>();
		for (TrainingData train : training) {
			double[] in = new double[numFeatures + 1];
			in[0] = 1.0;
			for (int j = 1; j < numFeatures + 1; j++) {
				in[j] = train.input[j - 1];
			}
			result.add(new TrainingData(in, train.output));
		}
		return result;
	}
	
	public static ArrayList<TrainingData> normalize(ArrayList<TrainingData> training){
		int numFeatures = training.get(0).input.length;
		//for each feature
		for(int i = 0; i < numFeatures; i++){
			//get all values, find max and min
			double MAX = Double.MIN_VALUE;
			double MIN = Double.MAX_VALUE;
			ArrayList<Double> allValues = new ArrayList<Double>();
			for(int j = 0; j < training.size(); j++){
				double val = training.get(j).input[i];
				if(val > MAX){
					MAX = val;
				}
				if(val < MIN){
					MIN = val;
				}
				allValues.add(val);
			}
			
			//okay now replace values with normalized
			for(int j = 0; j < training.size(); j++){
				double val = training.get(j).input[i];
				//gives range between 0 <-> 1
				val = (val - MIN) / (MAX - MIN);
				//range between 0 <-> 2
				val *= 2;
				//range between -1 <-> 1
				val -= 1.0;
				training.get(j).input[i] = val;
			}
		}
		return training;
	}
	
	public static ArrayList<TrainingData> readTrainingData(){
		ArrayList<TrainingData> list = new ArrayList<TrainingData>();
		try {
			Scanner s = new Scanner(new File(TRAINING_FILE));
			//read header to determine number of variables
			Scanner lineScanner = new Scanner(s.nextLine());
			int inputs = -1; //start at -1 to exclude last input, which is output
			while(lineScanner.hasNext()){
				inputs++;
				lineScanner.next();
			}
			lineScanner.close();
			s.nextLine();
			
			while(s.hasNextLine()){
				String line = s.nextLine();
				lineScanner = new Scanner(line);
				double[] input = new double[inputs];
				for(int i = 0; i < inputs; i++){
					input[i] = lineScanner.nextDouble();
				}
				list.add(new TrainingData(input, lineScanner.nextDouble()));
			}
			s.close();
		} catch (FileNotFoundException e) {
			System.err.println("File not Found");
			e.printStackTrace();
		}
		return list;
	}
	
}class TrainingData{
	public double[] input;
	public double output;
	public TrainingData(double[] i, double o){
		input = i;
		output = o;
	}
}
