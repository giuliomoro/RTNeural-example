#include "Plugin.h"
static StringArray model_json_array = { "conv.json", "dense.json", "full_model.json", "gru.json", "gru_1d.json", "lstm.json", "lstm_1d.json" };
//==============================================================================
RTNeuralExamplePlugin::RTNeuralExamplePlugin() :
#if JUCE_IOS || JUCE_MAC
    AudioProcessor (juce::JUCEApplicationBase::isStandaloneApp() ?
        BusesProperties().withInput ("Input", juce::AudioChannelSet::mono(), true)
                         .withOutput ("Output", juce::AudioChannelSet::stereo(), true) :
        BusesProperties().withInput ("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
#else
    AudioProcessor (BusesProperties().withInput ("Input", juce::AudioChannelSet::stereo(), true)
                                     .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
#endif
    parameters (*this, nullptr, Identifier ("Parameters"),
    {
        std::make_unique<AudioParameterFloat> ("gain_db", "Gain [dB]", -12.0f, 12.0f, 0.0f),
        std::make_unique<AudioParameterChoice> ("model_type", "Model Type", StringArray { "Run-Time", "Compile-Time" }, 0),
        std::make_unique<AudioParameterChoice> ("model_json", "Preset", model_json_array, 0),
        std::make_unique<AudioParameterBool> ("model_custom", "Load custom", true, ""),
    }),
    fileLoadListener(*this)
{
    inGainDbParam = parameters.getRawParameterValue ("gain_db");
    modelTypeParam = parameters.getRawParameterValue ("model_type");
    parameters.addParameterListener("model_type", &fileLoadListener);
    parameters.addParameterListener("model_custom", &fileLoadListener);
    parameters.addParameterListener("model_json", &fileLoadListener);

    MemoryInputStream jsonStream (BinaryData::neural_net_weights_json, BinaryData::neural_net_weights_jsonSize, false);
    auto jsonInput = nlohmann::json::parse (jsonStream.readEntireStreamAsString().toStdString());
    initRuntimeNN(jsonInput);

    neuralNetT[0].parseJson (jsonInput);
    neuralNetT[1].parseJson (jsonInput);
}

int RTNeuralExamplePlugin::initRuntimeNNFromFile(const File& file)
{
    int fail = 0;
    std::string s = file.loadFileAsString().toStdString();
    if ("" != s)
    {
        try
        {
            auto jsonInput = nlohmann::json::parse (s);
            fail = initRuntimeNN(jsonInput);
        }
        catch (std::exception& e)
        {
            fail = -1;
        }
    }
    else
        fail = -2;

    if(fail)
        AlertWindow::showMessageBoxAsync (AlertWindow::WarningIcon,
                                          TRANS("Error whilst loading"),
                                          TRANS("Couldn't read from the specified file, or file is not json, or it is incompatible json"));
    return fail;
}

int RTNeuralExamplePlugin::initRuntimeNN(const nlohmann::json& json)
{
    neuralNet[0] = RTNeural::json_parser::parseJson<float> (json);
    neuralNet[1] = RTNeural::json_parser::parseJson<float> (json);
    if(!neuralNet[0].get() || !neuralNet[1].get())
    {
        return 1;
    }
    neuralNet[0]->reset();
    neuralNet[1]->reset();
    return 0;
}

RTNeuralExamplePlugin::~RTNeuralExamplePlugin()
{
}

//==============================================================================
const String RTNeuralExamplePlugin::getName() const
{
    return JucePlugin_Name;
}

bool RTNeuralExamplePlugin::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool RTNeuralExamplePlugin::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool RTNeuralExamplePlugin::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double RTNeuralExamplePlugin::getTailLengthSeconds() const
{
    return 0.0;
}

int RTNeuralExamplePlugin::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int RTNeuralExamplePlugin::getCurrentProgram()
{
    return 0;
}

void RTNeuralExamplePlugin::setCurrentProgram (int index)
{
}

const String RTNeuralExamplePlugin::getProgramName (int index)
{
    return {};
}

void RTNeuralExamplePlugin::changeProgramName (int index, const String& newName)
{
}

//==============================================================================
void RTNeuralExamplePlugin::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    *dcBlocker.state = *dsp::IIR::Coefficients<float>::makeHighPass (sampleRate, 35.0f);

    dsp::ProcessSpec spec { sampleRate, static_cast<uint32> (samplesPerBlock), 2 };
    inputGain.prepare (spec);
    inputGain.setRampDurationSeconds (0.05);
    dcBlocker.prepare (spec);

    neuralNet[0]->reset();
    neuralNet[1]->reset();

    neuralNetT[0].reset();
    neuralNetT[1].reset();
}

void RTNeuralExamplePlugin::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

bool RTNeuralExamplePlugin::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    if (layouts.getMainOutputChannelSet() != AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;

    return true;
}

void RTNeuralExamplePlugin::processBlock (AudioBuffer<float>& buffer, MidiBuffer& midiMessages)
{
    ScopedNoDenormals noDenormals;

    dsp::AudioBlock<float> block (buffer);
    dsp::ProcessContextReplacing<float> context (block);

    inputGain.setGainDecibels (inGainDbParam->load() + 25.0f);
    inputGain.process (context);

    if(fileLoadedN.load() != fileToLoadN.load())
    {
        File file = fileToLoad;
        fileLoadedN.store(fileToLoadN.load());
        std::cout << "Loading: " << file.getFullPathName() << "\n";
        initRuntimeNNFromFile(file);
    }
    if (static_cast<int> (modelTypeParam->load()) == 0)
    {
        // use run-time model
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            auto* x = buffer.getWritePointer (ch);
            // make sure a valid NN is loaded before using it
            if(!neuralNet[ch].get())
                continue;
            for (int n = 0; n < buffer.getNumSamples(); ++n)
            {
                float input[] = { x[n] };
                x[n] = neuralNet[ch]->forward (input);
            }
        }
    }
    else
    {
        // use compile-time model
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            auto* x = buffer.getWritePointer (ch);
            for (int n = 0; n < buffer.getNumSamples(); ++n)
            {
                float input[] = { x[n] };
                x[n] = neuralNetT[ch].forward (input);
            }
        }
    }

    dcBlocker.process (context);
    buffer.applyGain (5.0f);

    ignoreUnused (midiMessages);
}

//==============================================================================
bool RTNeuralExamplePlugin::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

AudioProcessorEditor* RTNeuralExamplePlugin::createEditor()
{
    return new GenericAudioProcessorEditor (*this);
}

//==============================================================================
void RTNeuralExamplePlugin::getStateInformation (MemoryBlock& destData)
{
    auto state = parameters.copyState();
    std::unique_ptr<XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void RTNeuralExamplePlugin::setStateInformation (const void* data, int sizeInBytes)
{
    /*
    try {
        std::string s;
        const char* charData;
        for(int n = 0; n < sizeInBytes; ++n)
        {
            s += charData++;
        }
        printf("state: %*s\n", sizeInBytes, (const char*)data);
        auto jsonInput = nlohmann::json::parse (std::string((const char*)data));
        initRuntimeNN(jsonInput);
    } catch (std::exception&e) {
        std::cerr << "Invalid state passed in: \n";
        std::cerr << e.what();
    }
    */
    std::unique_ptr<XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
 
    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName (parameters.state.getType()))
            parameters.replaceState (ValueTree::fromXml (*xmlState));
}

void RTNeuralExamplePlugin::setNeuralNetFromJson (const void* data, int sizeInBytes)
{
}

//==============================================================================
// This creates new instances of the plugin..
AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new RTNeuralExamplePlugin();
}

void RTNeuralExamplePlugin::ButtonListener::parameterChanged(const String & parameter, float newValue)
{
    //auto checkbox = p.parameters.getParameter("model_custom");
    if (String("model_custom") == parameter)
    {
        FileChooser fc(TRANS("Load a json model"), File(), "*.json");
        if (fc.browseForFileToOpen())
        {
            p.queueNewRuntimeNN(fc.getResult());
            //checkbox->beginChangeGesture();
            //checkbox->setValueNotifyingHost(true);
            //checkbox->endChangeGesture();
        }
    } else if (String("model_json") == parameter)
    {
        String model = model_json_array[newValue];
        if (model != "")
        {
            String filename = "modules/RTNeural/models/" + model;
            p.queueNewRuntimeNN(filename);
            //checkbox->beginChangeGesture();
            //checkbox->setValueNotifyingHost(false);
            //checkbox->endChangeGesture();
        }
    }
}
