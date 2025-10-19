#include <NavierStokes/log.h>

#include <stdexcept>

template <typename T>
void csvPrintImpl(std::ostream & os, T && arg)
{
    os << std::forward<T>(arg);
}

template <typename T, typename... Args>
void csvPrintImpl(std::ostream & os, T && arg, Args &&... args)
{
    os << std::forward<T>(arg) << ',';
    csvPrintImpl(os, std::forward<Args>(args)...);
}

template <typename... Args>
void csvPrint(std::ostream & os, Args &&... args)
{
    if constexpr (sizeof...(args) > 0)
    {
        csvPrintImpl(os, std::forward<Args>(args)...);
    }
}

Log::Log(const std::string & fileName)
{
    file.open(fileName);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to create log file " + fileName);
    }

    file << "id,tTotal,tTentative,itersTentative,mseTentativeX,mseTentativeY,tPressure,itersPressure,msePressure,tFinal,itersFinal,mseFinalX,mseFinalY,cu_convection,cu_tentative,cu_pressure,cu_final\n";
}

void Log::add(const LogEntry & entry)
{
    csvPrint(file, entry.id, entry.tTotal,
             entry.tTentative, entry.itersTentative, entry.mseTentative[0], entry.mseTentative[1],
             entry.tPressure, entry.itersPressure, entry.msePressure,
             entry.tFinal, entry.itersFinal, entry.mseFinal[0], entry.mseFinal[1],
             entry.tCuConvection, entry.tCuTentative, entry.tCuPressure, entry.tCuFinal);
    file << "\n";
}