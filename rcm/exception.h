#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <stdexcept>
#include <string>

namespace rcm {

class system_error : public std::runtime_error
{
public:
	enum Reason
	{
		Negative_MC64_weight = -1,
		Matrix_singular      = -2
	};

	system_error(Reason             reason,
	             const std::string& what_arg)
	: std::runtime_error(what_arg),
	  m_reason(reason)
	{}

	system_error(Reason      reason,
	             const char* what_arg)
	: std::runtime_error(what_arg),
	  m_reason(reason)
	{}
	
	virtual ~system_error() throw() {}

	Reason  reason() const {return m_reason;}

private:
	Reason        m_reason;
};

} // namespace rcm


#endif
