import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import io
import base64

# Set parameters
mean = 10
std_dev = 3  # Chosen standard deviation for clean visual proportions
x_val = 4

# Generate normal distribution data
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
y = stats.norm.pdf(x, mean, std_dev)

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the main curve
ax.plot(x, y, color='black', linewidth=2)

# Highlight/color area to the left of x = 4
x_fill = np.linspace(mean - 4*std_dev, x_val, 500)
y_fill = stats.norm.pdf(x_fill, mean, std_dev)
ax.fill_between(x_fill, y_fill, color='#3498db', alpha=0.5, label='Area to the left of $x=4$')

# Draw vertical line at mean and x
ax.axvline(mean, color='darkgray', linestyle='--', linewidth=1.5)
ax.axvline(x_val, color='red', linestyle='-', linewidth=2)

# Add text labels on the axis
ax.set_xticks([x_val, mean])
ax.set_xticklabels([f'$x = {x_val}$', f'$\mu = {mean}$'], fontsize=12)

# Visual cleanups
ax.set_ylim(bottom=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_visible(False)

# Title
ax.set_title('Normal Distribution ($\mu = 10$)', fontsize=14, pad=15)

# Output logic
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
base64_str = base64.b64encode(buf.read()).decode('utf-8')
plt.close()
print(f'base64_encoded_image:"data:image/png;base64,{base64_str}"')
