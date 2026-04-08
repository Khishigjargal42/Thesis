FEATURE_TYPES = ['mel_spec', 'mfcc', 'spectrogram', 'log_mel']
results = {}

for feature_type in FEATURE_TYPES:
    print(f"\n{'='*40}")
    print(f"Training ResNet with: {feature_type}")
    
    # Чиний feature extraction функцуудаа дуудна
    X_train = load_features(feature_type, split='train')  # shape: (N, H, W, 1)
    X_val   = load_features(feature_type, split='val')
    X_test  = load_features(feature_type, split='test')
    
    # Model
    model = build_resnet(input_shape=X_train.shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Class weight (тэнцвэргүй өгөгдлийн тул)
    class_weight = {0: 1.0, 1: 0.26}  # 665/2575 ≈ 0.26
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    from sklearn.metrics import classification_report, roc_auc_score
    
    results[feature_type] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, model.predict(X_test))
    }
    print(classification_report(y_test, y_pred))

# Хүснэгт хэлбэрт гаргах
import pandas as pd
df = pd.DataFrame(results).T.round(4)
print(df)