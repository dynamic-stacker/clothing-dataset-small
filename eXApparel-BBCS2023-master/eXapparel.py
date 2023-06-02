

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'clothing-dataset-small-master/train'

# Define the image size and batch size
img_width, img_height = 150, 150
batch_size = 16

# Preprocess and augment the training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # Normalize pixel values to [0, 1]
    shear_range=0.2,            # Apply random shear transformations
    zoom_range=0.2,             # Apply random zoom transformations
    horizontal_flip=True       # Flip images horizontally
)

# Load and prepare the training dataset
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Create the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=1
)

# Save the trained model
model.save('clothing_classifier_model.h5')

# Prompt the user for clothing preferences
print("Select the clothing categories you are interested in:")
print("1. Dress")
print("2. Hat")
print("3. Longsleeve")
print("4. Outwear")
print("5. Pants")
print("6. Shirt")
print("7. Shoes")
print("8. Shorts")
print("9. Skirt")
print("10. T-shirt")

# Define the mapping of category numbers to category names
category_mapping = {
    1: 'dress',
    2: 'hat',
    3: 'longsleeve',
    4: 'outwear',
    5: 'pants',
    6: 'shirt',
    7: 'shoes',
    8: 'shorts',
    9: 'skirt',
    10: 'tshirt'
}

# Get user preferences
selected_categories = input("Enter the category numbers separated by commas (e.g., 1, 4, 7): ")
selected_categories = [int(category) for category in selected_categories.split(",")]

# Map the selected category numbers to category names
selected_category_names = [category_mapping[category] for category in selected_categories]

# Load the saved model
loaded_model = tf.keras.models.load_model('clothing_classifier_model.h5')

# Display the data from the desired categories
for category in selected_categories:
    category_name = list(train_generator.class_indices.keys())[category - 1]
    category_data_dir = f'/content/{train_data_dir}/{category_name}'

    # Load and prepare the data from the desired category
    category_datagen = ImageDataGenerator(rescale=1.0 / 255)
    category_generator = category_datagen.flow_from_directory(
        category_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )


  # Make predictions on the category data
    predictions = loaded_model.predict(category_generator)

    print(f"Category: {category_name}")
    for i in range(len(category_generator.filenames)):
        filename = category_generator.filenames[i]
        prediction = predictions[i]
        predicted_label = train_generator.class_indices[list(train_generator.class_indices.keys())[int(tf.argmax(prediction))]]

        print(f"Image: {filename}, Predicted Label: {predicted_label}")
