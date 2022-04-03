#pragma once

#include <JuceHeader.h>
#include <RTNeural/RTNeural.h>

class RTNeuralExamplePlugin  : public AudioProcessor
{
public:
    //==============================================================================
    RTNeuralExamplePlugin();
    ~RTNeuralExamplePlugin();

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (AudioBuffer<float>&, MidiBuffer&) override;

    //==============================================================================
    AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const String getProgramName (int index) override;
    void changeProgramName (int index, const String& newName) override;

    //==============================================================================
    void getStateInformation (MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;
    void setNeuralNetFromJson (const void* data, int sizeInBytes);

private:
    //==============================================================================
    AudioProcessorValueTreeState parameters;

    // input gain
    std::atomic<float>* inGainDbParam = nullptr;
    dsp::Gain<float> inputGain;
    
    // models
    std::atomic<float>* modelTypeParam = nullptr;
    int oldModel = -1;

    // example of model defined at run-time
    std::unique_ptr<RTNeural::Model<float>> neuralNet[2];

    // example of model defined at compile-time
    RTNeural::ModelT<float, 1, 1,
        RTNeural::DenseT<float, 1, 8>,
        RTNeural::TanhActivationT<float, 8>,
        RTNeural::Conv1DT<float, 8, 4, 3, 2>,
        RTNeural::TanhActivationT<float, 4>,
        RTNeural::GRULayerT<float, 4, 8>,
        RTNeural::DenseT<float, 8, 1>
    > neuralNetT[2];

    int initRuntimeNN(const nlohmann::json& json);
    int initRuntimeNNFromFile(const File& file);
    void queueNewRuntimeNN(const File& file)
    {
        printf("Queueing: %s\n", file.getFullPathName().toStdString().c_str());
        // lock-free queue, so the audio thread never blocks
        while(fileLoadedN.load() != fileToLoadN.load())
        {
            printf("waiting\n");
            Time::waitForMillisecondCounter(50);
        }
        fileToLoad = file;
        fileToLoadN.store(fileToLoadN.load() + 1);
    }

    dsp::ProcessorDuplicator<dsp::IIR::Filter<float>, dsp::IIR::Coefficients<float>> dcBlocker;

    class ButtonListener : public AudioProcessorValueTreeState::Listener {
    public:
        ButtonListener(RTNeuralExamplePlugin& p) : p(p) {}
        virtual void parameterChanged (const String & parameter, float newValue);
    private:
        RTNeuralExamplePlugin& p;
    };
    friend class ButtonListener;
    ButtonListener fileLoadListener;
    File fileToLoad;
    std::atomic<unsigned int> fileToLoadN = 0;
    std::atomic<unsigned int> fileLoadedN = 0;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (RTNeuralExamplePlugin)
};
