#include <SPI.h>
#include <TFT_eSPI.h> // Include the graphics library
#include <math.h>

TFT_eSPI tft = TFT_eSPI(); // Invoke custom library instance

#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 240

// --- Stanza 1 Vars ---
float plasmaTime = 0.0;
float wanderX, wanderY;
float wanderAngle = 0.0;
float wanderA = 80.0, wanderB = 80.0; // Lissajous params
float wanderFreqA = 1.0, wanderFreqB = 1.3; // Lissajous frequencies

// --- Stanza 2 Vars ---
unsigned long lastFlashTime = 0;
int flashDuration = 150; // ms
bool showFlash = false;
int dissolveRate = 50; // How many random pixels to clear per frame

// --- Stanza 3 Vars ---
int numChains = 0;
int maxChains = 30;
struct Line {
    int x1, y1, x2, y2;
    uint16_t color;
};
Line chains[maxChains];
unsigned long lastChainAddTime = 0;
int chainAddInterval = 500; // ms

// --- General ---
enum DemoPhase { STANZA1, STANZA2, STANZA3 };
DemoPhase currentPhase = STANZA1;
unsigned long phaseStartTime = 0;
unsigned long stanzaDuration = 20000; // 20 seconds per stanza (adjust as needed)

// --- Helper Functions ---

// Simple Plasma Calculation (adjust for desired look/speed)
float calculatePlasma(float x, float y, float t) {
    float val = 0.0;
    val += sin(x * 0.05 + t * 0.5);
    val += sin(y * 0.03 + t * 0.3);
    val += sin(sqrt((x - SCREEN_WIDTH / 2.0) * (x - SCREEN_WIDTH / 2.0) + (y - SCREEN_HEIGHT / 2.0) * (y - SCREEN_HEIGHT / 2.0)) * 0.02 + t);
    val += sin(sqrt(x * x + y * y) * 0.02 + t * 0.8);
    return val / 4.0; // Normalize
}

// Map plasma value to color based on phase
uint16_t plasmaColor(float value, DemoPhase phase) {
    // Map value from [-1, 1] potentially to [0, 1] first if needed
    float normValue = (value + 1.0) / 2.0; // Normalize to [0, 1]
    uint8_t r, g, b;

    switch (phase) {
        case STANZA1: // Spring, light: Greens, Yellows, Light Blues
            r = (uint8_t)(100 + 100 * sin(normValue * 3.14 * 2.0 + 1.0));
            g = (uint8_t)(150 + 100 * sin(normValue * 3.14 * 2.0 + 2.0));
            b = (uint8_t)(100 + 100 * sin(normValue * 3.14 * 2.0 + 0.0));
             // Emphasize green/yellow
            if (g < 128) g = 128 + random(0,50) ;
            if (r < 50) r = 50+ random(0,50);
            if (b > 200) b = 200;
            break;
        case STANZA2: // Dissolving, uncertain: Muted, Grays, flashes
             r = (uint8_t)(100 + 80 * sin(normValue * 3.14 * 3.0 + 1.5));
             g = (uint8_t)(100 + 80 * sin(normValue * 3.14 * 3.0 + 1.5));
             b = (uint8_t)(120 + 90 * sin(normValue * 3.14 * 3.0 + 1.0));
            break;
        case STANZA3: // Constrained, sad: Dark Blues, Purples, Grays
            r = (uint8_t)(50 + 50 * sin(normValue * 3.14 * 4.0 + 3.0));
            g = (uint8_t)(50 + 50 * sin(normValue * 3.14 * 4.0 + 3.0));
            b = (uint8_t)(100 + 100 * sin(normValue * 3.14 * 4.0 + 2.5));
             // Emphasize blue/purple
            if (b < 100) b = 100+random(0,50);
            if (r > 150) r = 150;
            if (g > 150) g = 150;
            break;
    }
    return tft.color565(r, g, b);
}

void drawPlasmaBackground(DemoPhase phase) {
    // Optimization: Draw plasma in blocks or lines to speed up
    // This is a basic pixel-by-pixel version - might be slow
    for (int y = 0; y < SCREEN_HEIGHT; y += 4) { // Step by 4 for speed
        for (int x = 0; x < SCREEN_WIDTH; x += 4) { // Step by 4 for speed
            float value = calculatePlasma(x, y, plasmaTime);
            uint16_t color = plasmaColor(value, phase);
            tft.fillRect(x, y, 4, 4, color);
        }
    }
    plasmaTime += 0.02; // Increment time for evolution
}

void drawWanderingSoul(DemoPhase phase) {
    // Calculate Lissajous position
    wanderX = SCREEN_WIDTH / 2.0 + wanderA * sin(wanderFreqA * wanderAngle);
    wanderY = SCREEN_HEIGHT / 2.0 + wanderB * cos(wanderFreqB * wanderAngle); // Use cos for variation
    wanderAngle += 0.03;

    uint16_t color = TFT_WHITE;
    int size = 3;

    if (phase == STANZA2) {
        // Make it intermittent/dashed
        if (millis() % 500 < 250) {
             color = plasmaColor(calculatePlasma(wanderX, wanderY, plasmaTime), phase); // Blend with bg
             // color = TFT_DARKGREY;
             size = 2;
        } else {
          return; // Skip drawing sometimes
        }
    } else if (phase == STANZA3) {
        // Make it fainter or stop
         color = TFT_NAVY; // Darker color
         size = 1;
         // Maybe freeze it: wanderAngle doesn't increment here or increments slower
    }

    tft.fillCircle(wanderX, wanderY, size, color);
}


void drawDissolveEffect() {
     // Draw random black/background pixels to "dissolve"
    for(int i=0; i < dissolveRate; ++i) {
        int rx = random(0, SCREEN_WIDTH);
        int ry = random(0, SCREEN_HEIGHT);
        // Get bg color at that point to blend better
        float value = calculatePlasma(rx, ry, plasmaTime);
        uint16_t bgColor = plasmaColor(value, currentPhase);
        tft.drawPixel(rx, ry, bgColor);
         //tft.drawPixel(random(0, SCREEN_WIDTH), random(0, SCREEN_HEIGHT), TFT_BLACK);
    }
}

void drawFlashEffect() {
    if (showFlash) {
        tft.fillScreen(TFT_WHITE); // Simple bright flash
        if (millis() - lastFlashTime > flashDuration) {
            showFlash = false;
            // Need to redraw background after flash ends
             tft.fillScreen(TFT_BLACK); // Clear before redraw
             drawPlasmaBackground(currentPhase);
        }
    } else {
        // Trigger flash occasionally
        if (random(0, 400) == 0) { // Low probability each frame
            showFlash = true;
            lastFlashTime = millis();
        }
    }
}

void addChain() {
     if (numChains < maxChains) {
        chains[numChains].x1 = random(0, SCREEN_WIDTH);
        chains[numChains].y1 = random(0, SCREEN_HEIGHT);
        chains[numChains].x2 = random(0, SCREEN_WIDTH);
        chains[numChains].y2 = random(0, SCREEN_HEIGHT);
        // Darker, constraining colors
        uint8_t grey = random(50, 150);
        chains[numChains].color = tft.color565(grey, grey, grey+random(0,50)); // Greyish-blue
        numChains++;
        lastChainAddTime = millis();
    }
}

void drawChains() {
    for (int i = 0; i < numChains; i++) {
        tft.drawLine(chains[i].x1, chains[i].y1, chains[i].x2, chains[i].y2, chains[i].color);
    }
    // Add new chains periodically
    if (millis() - lastChainAddTime > chainAddInterval) {
         addChain();
         // Optionally speed up adding chains over time
         // if (chainAddInterval > 100) chainAddInterval -= 10;
    }
}

void displayTextForPhase(DemoPhase phase) {
    tft.setTextSize(2);
    tft.setTextDatum(MC_DATUM); // Middle Center datum

    unsigned long timeInPhase = millis() - phaseStartTime;
    int fadeDuration = 2000; // 2 seconds fade in/out
    uint8_t alpha = 255; // Max brightness (Note: TFT_eSPI doesn't directly support alpha blending easily for fonts)
                         // We simulate fading by changing color towards background

    // Simple "fade" by changing color brightness or choosing text color
    uint16_t textColor = TFT_WHITE;
    uint16_t bgColor = TFT_BLACK; // Assume black bg for text area

    if (phase == STANZA2 && timeInPhase < stanzaDuration / 2) { // Only show for first half
        textColor = TFT_LIGHTGREY;
        if (timeInPhase < fadeDuration) { // Fade in
             // Simple brightness fade doesn't work well without alpha, let's just show it
        } else if (stanzaDuration / 2 - timeInPhase < fadeDuration) { // Fade out
            textColor = TFT_DARKGREY; // Make it darker to simulate fade
        }
         tft.setTextColor(textColor, bgColor);
         tft.drawString("Les mots se dissolvent...", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 10);
    } else if (phase == STANZA3 && timeInPhase > fadeDuration) { // Show after a delay
        textColor = TFT_NAVY; // Darker text
        tft.setTextColor(textColor, bgColor);
        tft.drawString("...l'enchaîner...", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 10);
    }
    // Stanza 1 text could be added similarly if desired
}


// --- Arduino Setup & Loop ---

void setup() {
    Serial.begin(115200);
    randomSeed(analogRead(0));

    tft.init();
    tft.setRotation(1); // Adjust rotation if needed (0, 1, 2, 3)
    tft.fillScreen(TFT_BLACK);

    phaseStartTime = millis();
    currentPhase = STANZA1;

     // Pre-calculate some chain positions if needed for startup
    // addChain();
}

void loop() {
    unsigned long currentTime = millis();

    // --- Phase Transition Logic ---
    if (currentTime - phaseStartTime > stanzaDuration) {
        phaseStartTime = currentTime; // Reset timer for the new phase
        switch (currentPhase) {
            case STANZA1:
                currentPhase = STANZA2;
                tft.fillScreen(TFT_BLACK); // Clear screen for transition effect
                break;
            case STANZA2:
                currentPhase = STANZA3;
                tft.fillScreen(TFT_BLACK);
                 // Reset chain parameters for stanza 3 start
                 numChains = 0;
                 lastChainAddTime = millis();
                break;
            case STANZA3:
                currentPhase = STANZA1; // Loop back
                tft.fillScreen(TFT_BLACK);
                // Reset parameters for stanza 1 if needed
                 wanderAngle = 0;
                break;
        }
    }

    // --- Drawing Logic based on Phase ---

    // Always draw background (or handle flash)
    if (currentPhase == STANZA2 && showFlash) {
         drawFlashEffect(); // Handles its own timing and background redraw needs
    } else {
         drawPlasmaBackground(currentPhase);
    }


    // Stanza-specific elements (only if not flashing)
     if (!(currentPhase == STANZA2 && showFlash)) {
        switch (currentPhase) {
            case STANZA1:
                drawWanderingSoul(currentPhase);
                break;
            case STANZA2:
                drawWanderingSoul(currentPhase);
                drawDissolveEffect();
                // Flash is handled above, text is drawn below
                break;
            case STANZA3:
                 drawWanderingSoul(currentPhase); // Draw faint/constrained soul
                 drawChains(); // Draw the geometric constraints
                break;
        }
        // Display text overlay (optional) - might need a background box for clarity
        // displayTextForPhase(currentPhase); // Uncomment if text is desired
     }


    // Small delay to prevent watchdog timer issues and control speed
    delay(10);
}
