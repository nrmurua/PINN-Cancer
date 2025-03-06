def test_with_init_forward(model, printable=False):
    try:
        output_test = model.init_forward()
        if printable:
            print(output_test)
    except Exception as e:
        print(f"An error occurred: {e}")

    