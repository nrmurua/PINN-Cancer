from plots import plot2D

def test_with_init_forward(model, printable=False):
    try:
        output_test = model.init_forward()
        if printable:
            print(output_test)
        print("PINN is working")
    except Exception as e:
        print(f"An error occurred: {e}")

def test_plot2D(N, T, I, x, t):
    try:
        N_plot = N.squeeze(0).cpu().numpy()
        T_plot = T.squeeze(0).cpu().numpy()
        I_plot = I.squeeze(0).cpu().numpy()

        plot2D(N_plot, x, t)
        plot2D(T_plot, x, t)
        plot2D(I_plot, x, t)
        
        print("Plots 2D are being generated correctly")
    except Exception as e:
        print(f"An error occurred: {e}")