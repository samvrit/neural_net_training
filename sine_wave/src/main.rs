use ndarray::Array2;
use rand::Rng;
use plotters::prelude::*;

const LEARNING_RATE: f64 = 0.005;
const EPOCHS: usize = 100000;

const INPUT_SIZE: usize = 1;
const HIDDEN1_SIZE: usize = 16;
const OUTPUT_SIZE: usize = 1;

const PI: f64 = std::f64::consts::PI;

// Inline ReLU activation for better performance
#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

// Inline ReLU derivative for better performance
#[inline]
fn relu_derivative(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();

    // Pre-allocate weight matrices with optimized initialization
    let mut w1: Array2<f64> = Array2::from_shape_fn((INPUT_SIZE, HIDDEN1_SIZE), |_| rng.gen_range(-0.1..0.1));
    let mut w2: Array2<f64> = Array2::from_shape_fn((HIDDEN1_SIZE, HIDDEN1_SIZE), |_| rng.gen_range(-0.1..0.1));
    let mut w3: Array2<f64> = Array2::from_shape_fn((HIDDEN1_SIZE, OUTPUT_SIZE), |_| rng.gen_range(-0.1..0.1));
    
    // Bias vectors initialized to zero
    let mut b1: Array2<f64> = Array2::zeros((1, HIDDEN1_SIZE));
    let mut b2: Array2<f64> = Array2::zeros((1, HIDDEN1_SIZE));
    let mut b3: Array2<f64> = Array2::zeros((1, OUTPUT_SIZE));

    // Pre-allocate loss vector and intermediate arrays for forward pass
    let mut losses: Vec<f64> = Vec::with_capacity(EPOCHS);
    let mut l1_store = Array2::zeros((1, HIDDEN1_SIZE));
    let mut y1_store = Array2::zeros((1, HIDDEN1_SIZE));
    let mut l2_store = Array2::zeros((1, HIDDEN1_SIZE));
    let mut y2_store = Array2::zeros((1, HIDDEN1_SIZE));
    
    // Gradient storage to reduce allocations
    let mut w3_grad = Array2::zeros((HIDDEN1_SIZE, OUTPUT_SIZE));
    let mut w2_grad = Array2::zeros((HIDDEN1_SIZE, HIDDEN1_SIZE));
    let mut w1_grad = Array2::zeros((INPUT_SIZE, HIDDEN1_SIZE));
    let mut b3_grad = Array2::zeros((1, OUTPUT_SIZE));
    let mut b2_grad = Array2::zeros((1, HIDDEN1_SIZE));
    let mut b1_grad = Array2::zeros((1, HIDDEN1_SIZE));

    for _ in 0..EPOCHS {
        let input = rng.gen_range(0.0..2.0*PI);
        let target = input.sin();

        // Forward pass with in-place operations where possible
        l1_store.assign(&(&w1 * input + &b1));
        y1_store.assign(&l1_store.mapv(relu));
        
        l2_store.assign(&(y1_store.dot(&w2) + &b2));
        y2_store.assign(&l2_store.mapv(relu));
        
        let y3 = y2_store.dot(&w3) + &b3;
        let output = y3[[0, 0]];
        
        let loss = (output - target).powi(2);
        losses.push(loss);

        // Backward pass with optimized gradient computation
        let loss_derivative = 2.0 * (output - target);
        
        // Gradients for layer 3
        b3_grad.fill(loss_derivative);
        w3_grad.assign(&(&y2_store.t().to_owned() * loss_derivative));

        // Gradients for layer 2
        let mut loss_to_l2_derivative = w3.t().to_owned() * loss_derivative;
        loss_to_l2_derivative *= &l2_store.mapv(relu_derivative);
        b2_grad.assign(&loss_to_l2_derivative);
        w2_grad.assign(&y1_store.t().dot(&loss_to_l2_derivative));

        // Gradients for layer 1
        let mut loss_to_l1_derivative = loss_to_l2_derivative.dot(&w2.t());
        loss_to_l1_derivative *= &l1_store.mapv(relu_derivative);
        b1_grad.assign(&loss_to_l1_derivative);
        w1_grad.assign(&(&loss_to_l1_derivative * input));

        // Update weights and biases with optimized operations
        w3.scaled_add(-LEARNING_RATE, &w3_grad);
        b3.scaled_add(-LEARNING_RATE, &b3_grad);
        w2.scaled_add(-LEARNING_RATE, &w2_grad);
        b2.scaled_add(-LEARNING_RATE, &b2_grad);
        w1.scaled_add(-LEARNING_RATE, &w1_grad);
        b1.scaled_add(-LEARNING_RATE, &b1_grad);
    }

    // Visualize training loss
    if !losses.is_empty() {
        let max_loss = losses.iter().copied().fold(0.0, f64::max);
        let root = BitMapBackend::new("images/loss.png", (800, 600)).into_drawing_area();
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

    // Evaluate network on dense grid using pre-allocated buffers
    const POINTS: usize = 1000;
    let xs: Vec<f64> = (0..POINTS)
        .map(|i| i as f64 * 2.0 * PI / ((POINTS - 1) as f64))
        .collect();
    
    let mut preds: Vec<f64> = Vec::with_capacity(POINTS);
    let mut eval_l1 = Array2::zeros((1, HIDDEN1_SIZE));
    let mut eval_y1 = Array2::zeros((1, HIDDEN1_SIZE));
    let mut eval_l2 = Array2::zeros((1, HIDDEN1_SIZE));
    let mut eval_y2 = Array2::zeros((1, HIDDEN1_SIZE));
    
    for &x in &xs {
        eval_l1.assign(&(&w1 * x + &b1));
        eval_y1.assign(&eval_l1.mapv(relu));
        eval_l2.assign(&(eval_y1.dot(&w2) + &b2));
        eval_y2.assign(&eval_l2.mapv(relu));
        let eval_y3 = eval_y2.dot(&w3) + &b3;
        preds.push(eval_y3[[0, 0]]);
    }
    
    let trues: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();

    // Plot predictions vs actual sine
    if !xs.is_empty() {
        let y_min = trues
            .iter()
            .chain(preds.iter())
            .copied()
            .fold(f64::INFINITY, f64::min);
        let y_max = trues
            .iter()
            .chain(preds.iter())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let root2 = BitMapBackend::new("images/prediction.png", (800, 600)).into_drawing_area();
        root2.fill(&WHITE)?;
        let mut chart2 = ChartBuilder::on(&root2)
            .caption("Network vs sin(x)", ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0..(POINTS as i32), y_min..y_max)?;
        chart2.configure_mesh().draw()?;
        chart2.draw_series(LineSeries::new(
            xs.iter().enumerate().map(|(i, &_x)| (i as i32, trues[i])),
            &BLUE,
        ))?
        .label("sin(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart2.draw_series(LineSeries::new(
            xs.iter().enumerate().map(|(i, &_x)| (i as i32, preds[i])),
            &RED,
        ))?
        .label("network")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart2.configure_series_labels().border_style(&BLACK).draw()?;
    }

    // Display final trained weights and biases
    println!("final w1:\n{}", w1);
    println!("final w2:\n{}", w2);
    println!("final w3:\n{}", w3);
    println!("final b1:\n{}", b1);
    println!("final b2:\n{}", b2);
    println!("final b3:\n{}", b3);
    
    Ok(())
}
