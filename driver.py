from functions import *
import tensorflow as tf

if __name__ == "__main__":
    
    # Create training/testing data
    create_data()
    
    # Create figure 2
    create_data_example()
    
    # Values for table 3
    print("Training:")
    positive, negative = dataset_split("/gws/nopw/j04/hiresgw/dg/paper_1_ibtracs_filtered", [1980, 1981, 1982, 1983, 1984, 1985, 1987, 1988, 1989, 1990, 1992, 1993, 1994, 1995, 1997, 1998, 1999, 2000, 2002, 2003, 2004, 2005, 2007, 2008, 2009, 2010, 2012, 2013, 2014, 2015, 2017])
    print("Positive:", positive, "Negative:", negative)
    
    print("Validation:")
    positive, negative = dataset_split("/gws/nopw/j04/hiresgw/dg/paper_1_ibtracs_filtered", [1979, 1986, 1991, 1996, 2001, 2006, 2011, 2016])
    print("Positive:", positive, "Negative:", negative)
    
    print("Testing:")
    positive, negative = dataset_split("/gws/nopw/j04/hiresgw/dg/paper_1_ibtracs_test_filtered", [2017, 2018, 2019])
    print("Positive:", positive, "Negative:", negative)
    
    # Load test data
    train_data, train_labels = load_dl_data("/gws/nopw/j04/hiresgw/dg/paper_1_ibtracs_filtered", [1980, 1981, 1982, 1983, 1984, 1985, 1987, 1988, 1989, 1990, 1992, 1993, 1994, 1995, 1997, 1998, 1999, 2000, 2002, 2003, 2004, 2005, 2007, 2008, 2009, 2010, 2012, 2013, 2014, 2015, 2017])
    val_data, val_labels = load_dl_data("/gws/nopw/j04/hiresgw/dg/paper_1_ibtracs_filtered", [1979, 1986, 1991, 1996, 2001, 2006, 2011, 2016])
    
    # Train DL model
    model = train_dl_model(train_data, train_labels, val_data, val_labels, "model")
    
    # Load DL model
    model = tf.keras.models.load_model("final_whole_world")
    
    # Load test data
    data, labels, files = load_dl_data("/gws/nopw/j04/hiresgw/dg/paper_1_ibtracs_test_filtered")
    
    # Get confusion matrix of table 4
    get_conf_matrix(data, labels, model)
    
    # Get AUC-PR curve of figure 4
    get_aucpr_curve(model, data, labels)
    
    # Compare standard models
    imagenet_comp("imagenet_tests.pdf")
    
    # Perform feature importance and obtain figure 6
    feature_importance(model, data, labels, 30)
    
    # Get recall per maximum category present in case, to get table 6
    recall_by_cat(model, "/gws/nopw/j04/hiresgw/dg/paper_1_ibtracs_test_filtered")
    
    # Split positively inferred cases by max cat present in case
    positive_cases_by_cat(model, data, files)
    
    # Show that WAWP model doesn't work as well on global data (table 9)
    wawp_model = tf.keras.models.load_model("final_wawp")
    wawp_data, wawp_labels, wawp_files = load_dl_data("/gws/nopw/j04/hiresgw/dg/paper_1_ibtracs_test_filtered", whole_world=False)
    _, _, _, _, _, _, _, _, auc, _ = eval_model(wawp_model, wawp_data, wawp_labels)
    print("AUC-PR for WAWP model tested on WAWP data:", auc)
    _, _, _, _, _, _, _, _, auc, _ = eval_model(wawp_model, data, labels)
    print("AUC-PR for WAWP model tested on global data:", auc)
    _, _, _, _, _, _, _, _, auc, _ = eval_model(model, data, labels)
    print("AUC-PR for global model tested on global data:", auc)
    
    # Get the recall of actual TCs per basin for both the WAWP model and global model (table 10)
    recall_by_basin(wawp_model, model, data, labels, files)
    
    # Get the mean state of the WAWP vs global cases (figure 7)
    get_mean_data(wawp_data, data)
    
    # Plot AUC-PR vs dataset size in % (figure 8)
    dataset_size()