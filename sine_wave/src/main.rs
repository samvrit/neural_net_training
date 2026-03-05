use ndarray::Array2;
use rand::Rng; // Import the Rng trait to use gen_range
use plotters::prelude::*;

const LEARNING_RATE: f64 = 0.005;
const EPOCHS: usize = 100000;

const PI: f64 = std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    let mut input: f64; // input will be set each epoch

    // create mutable weight matrices initialized randomly between 0 and 1
    let mut w1: Array2<f64> = Array2::from_shape_fn((1, 16), |_| rng.gen_range(-0.1..0.1));
    let mut w2: Array2<f64> = Array2::from_shape_fn((16, 16), |_| rng.gen_range(-0.1..0.1));
    let mut w3: Array2<f64> = Array2::from_shape_fn((16, 1), |_| rng.gen_range(-0.1..0.1));
    // biases remain initialized to ones for now
    let mut b1: Array2<f64> = Array2::zeros((1, 16));
    let mut b2: Array2<f64> = Array2::zeros((1, 16));
    let mut b3: Array2<f64> = Array2::zeros((1, 1));

    let mut w1_gradients: Array2<f64> = Array2::zeros((1, 16));
    let mut w2_gradients: Array2<f64> = Array2::zeros((16, 16));
    let mut w3_gradients: Array2<f64> = Array2::zeros((16, 1));
    let mut b1_gradients: Array2<f64> = Array2::zeros((1, 16));
    let mut b2_gradients: Array2<f64> = Array2::zeros((1, 16));
    let mut b3_gradients: Array2<f64> = Array2::zeros((1, 1));

    let mut l1: Array2<f64> = Array2::zeros((1, 16));
    let mut l2: Array2<f64> = Array2::zeros((1, 16));
    let mut l3: Array2<f64> = Array2::zeros((1, 1));
    let mut y1: Array2<f64> = Array2::zeros((1, 16));
    let mut y2: Array2<f64> = Array2::zeros((1, 16));
    let mut y3: Array2<f64> = Array2::zeros((1, 1));

    // keep track of loss values for visualization
    let mut losses: Vec<f64> = Vec::with_capacity(EPOCHS);

    for _ in 0..EPOCHS {
        input = rng.gen_range(0.0..2.0*PI); // Random value between 0 and π

        // Forward pass
        // multiply each weight by the input scalar without taking ownership
        l1 = w1.mapv(|x| x * input) + &b1;
        y1 = l1.mapv(|x| x.max(0.0)); // ReLU activation
        l2 = y1.dot(&w2) + &b2;
        y2 = l2.mapv(|x| x.max(0.0)); // ReLU activation
        l3 = y2.dot(&w3) + &b3;
        y3 = l3; // No activation for the output layer

        let loss = (y3[(0, 0)] - input.sin()).powi(2);
        losses.push(loss);

        // Backward pass (compute gradients)
        let loss_derivative = 2.0 * (y3[(0, 0)] - input.sin());
        let dy2_dl2 = l2.mapv(|x| if x >= 0.0 { 1.0 } else { 0.0 }); // ReLU derivative mask for l2
        let dy1_dl1 = l1.mapv(|x| if x >= 0.0 { 1.0 } else { 0.0 }); // ReLU derivative mask for l1
        
        let loss_to_l3_derivative = loss_derivative; // since l3 is the output layer
        w3_gradients = y2.t().to_owned() * loss_to_l3_derivative;
        b3_gradients = Array2::from_elem((1, 1), loss_to_l3_derivative);

        // multiply scalar by transposed w3, then apply ReLU derivative mask
        let loss_to_l2_derivative = w3.t().to_owned().mapv(|x| x * loss_to_l3_derivative) * &dy2_dl2;
        // gradient of w2: y1^T (16×1) · dL/dl2 (1×16) → 16×16
        w2_gradients = y1.t().dot(&loss_to_l2_derivative);
        b2_gradients = loss_to_l2_derivative.clone();

        let loss_to_l1_derivative = loss_to_l2_derivative.dot(&w2.t()) * &dy1_dl1;
        w1_gradients = loss_to_l1_derivative.clone() * input;
        b1_gradients = loss_to_l1_derivative.clone();

        // Update weights and biases
        w3 = &w3 - &(LEARNING_RATE * &w3_gradients);
        b3 = &b3 - &(LEARNING_RATE * &b3_gradients);
        w2 = &w2 - &(LEARNING_RATE * &w2_gradients);
        b2 = &b2 - &(LEARNING_RATE * &b2_gradients);
        w1 = &w1 - &(LEARNING_RATE * &w1_gradients);
        b1 = &b1 - &(LEARNING_RATE * &b1_gradients);
    }

    // create a PNG showing loss over iterations
    if !losses.is_empty() {
        let max_loss = losses.iter().cloned().fold(0.0, f64::max);
        let root = BitMapBackend::new("loss.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("Training Loss", ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0..(EPOCHS as i32), 0.0..max_loss)?;

        chart.configure_mesh().draw()?;
        chart.draw_series(LineSeries::new(
            losses.iter().enumerate().map(|(i, &l)| (i as i32, l)),
            &RED,
        ))?;
    }

    // after training, sample the network across [0, 2π]
    const POINTS: usize = 1000;
    let xs: Vec<f64> = (0..POINTS)
        .map(|i| i as f64 * 2.0 * PI / ((POINTS - 1) as f64))
        .collect();
    let mut preds: Vec<f64> = Vec::with_capacity(POINTS);
    for &x in &xs {
        let l1 = w1.mapv(|a| a * x) + &b1;
        let y1 = l1.mapv(|v| v.max(0.0));
        let l2 = y1.dot(&w2) + &b2;
        let y2 = l2.mapv(|v| v.max(0.0));
        let l3 = y2.dot(&w3) + &b3;
        preds.push(l3[(0, 0)]);
    }
    let trues: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();

    // plot predictions vs actual sine
    if !xs.is_empty() {
        let y_min = trues
            .iter()
            .chain(preds.iter())
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let y_max = trues
            .iter()
            .chain(preds.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let root2 = BitMapBackend::new("prediction.png", (800, 600)).into_drawing_area();
        root2.fill(&WHITE)?;
        let mut chart2 = ChartBuilder::on(&root2)
            .caption("Network vs sin(x)", ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0..(POINTS as i32), y_min..y_max)?;
        chart2.configure_mesh().draw()?;
        chart2.draw_series(LineSeries::new(
            xs.iter().enumerate().map(|(i, &x)| (i as i32, trues[i])),
            &BLUE,
        ))?
        .label("sin(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart2.draw_series(LineSeries::new(
            xs.iter().enumerate().map(|(i, &x)| (i as i32, preds[i])),
            &RED,
        ))?
        .label("network")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart2.configure_series_labels().border_style(&BLACK).draw()?;
    }

    // display final trained weights and biases (not gradients)
    println!("final w1: {}", w1);
    println!("final w2: {}", w2);
    println!("final w3: {}", w3);
    println!("final b1: {}", b1);
    println!("final b2: {}", b2);
    println!("final b3: {}", b3);
    
    Ok(())
}
