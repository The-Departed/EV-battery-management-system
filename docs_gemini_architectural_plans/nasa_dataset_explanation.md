# Why We Separate Charging and Driving Data

You asked a very good, highly technical question: *"Why do we only take charging data for SOH? Shouldn't we monitor SOH all the time while driving?"*

This is the exact question a professor will ask to test if you actually understand battery chemistry, or if you just copy-pasted code. Here is the undeniable scientific reason why the world's best EV engineers split the data exactly how we are doing it.

---

## 1. Why SOH is Calculated ONLY During Charging

**The Short Answer:** Driving is too chaotic. Charging is perfectly controlled.

**The Detailed Engineering Answer:**
Imagine trying to determine exactly how many liters of water a bucket can hold. 
*   **While Driving (Discharging):** It is like someone repeatedly kicking the bucket, aggressively sloshing water out in random bursts, then stopping, then kicking it again. You cannot accurately measure the total capacity of the bucket while it is experiencing extreme, unpredictable turbulence. In an EV, pressing the accelerator pulls 200 Amps one second, then drops to 0 Amps, then regenerative braking pushes 50 Amps back in. The voltage is bouncing violently up and down. You cannot extract reliable long-term degradation math from chaos.
*   **While Charging:** It is like turning on a smooth, perfectly calibrated faucet. The EV is plugged into the wall. The charger provides a perfectly steady, constant 1.5 Amps (called Constant Current or CC charging). Because the incoming energy is perfectly smooth and identical every single night, any change in how the battery reacts is **100% due to aging**, not due to how you drove the car.

**Conclusion:** We use the LSTM to look at the *Charging Phase* because it is a clean, controlled laboratory environment where the AI can perfectly spot the tiny, subtle signs of a dying battery over 160 days.

---

## 2. Why We Use the "Other Half" for the Transformer

If the AI calculates that the battery is old (SOH = 80%) during the overnight charge, we don't just throw that information away. We *use* it during the drive.

*   The **LSTM** calculates the SOH while you sleep.
*   When you wake up and drive to work, the **Transformer** says: *"Okay, the LSTM told me this battery is old and highly resistant. Therefore, when the driver hits the gas pedal right now, I know the Core Temperature is going to spike much faster than it did last year."*

We use the other half of the dataset (The Driving Cycles) precisely to train the Transformer on how to deal with the chaos of highway driving, armed with the knowledge of battery age that the LSTM provided.

---

## 3. What Does the NASA Dataset Actually Look Like?

When we download NASA's `B0005.mat` file, it is not just one giant spreadsheet. It is a timeline of events that lasted for months. It looks like a giant dictionary list:

### Event 1: The "Charge" Cycle
NASA plugged the battery into a wall charger.
*   **Time:** 0 sec -> 9,000 sec
*   **Voltage:** Slowly smoothly rises from 3.2V to 4.2V
*   **Current:** Held at roughly 1.5 Amps steady.
*   **Temperature:** Slightly warm.
*   *👉 This is what we feed to our Phase A LSTM.*

### Event 2: The "Discharge" (Drive) Cycle
NASA disconnected the charger and put a massive, random electrical load on the battery to simulate driving.
*   **Time:** 0 sec -> 3,000 sec
*   **Voltage:** Jaggedly crashing and spiking between 4.2V and 2.7V.
*   **Current:** Violently jumping between -4 Amps and 0 Amps.
*   **Temperature:** Getting dangerously hot.
*   *👉 This is what we feed to our Digital Twin Physics Engine to calculate Core Temp, and then to our Phase C Transformer.*

### Event 3: The "Impedance" (Lab Test) Cycle
NASA put the battery in an ultra-precise EIS machine to measure internal resistance. 
*   *👉 We use this data as the "Truth Label" to prove our LSTM is actually correct about the battery aging.*

*(This cycle repeats 168 times over the course of the battery's lifespan, until it finally dies)*

---

### In Summary for your Viva:
*"We track SOH during charging because Constant Current profiles provide a noise-free baseline to observe irreversible capacity fade. We then pass that SOH state to the Transformer during the discharging cycle, so the thermal prediction dynamically adapts to the increased internal resistance of an aged cell."*

Does this clarify exactly *why* we cut the data in half, what each half is shaped like, and why this proves you understand EV battery physics on a deep level?
